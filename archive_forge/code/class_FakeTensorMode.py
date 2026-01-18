import contextlib
import functools
import itertools
import logging
import os
import sys
import traceback
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union
from weakref import ReferenceType
import torch
import torch._custom_op
import torch._logging
from torch._guards import Source
from torch._ops import OpOverload
from torch._prims_common import (
from torch._subclasses.meta_utils import MetaConverter
from torch._utils import render_call
from torch.fx.operator_schemas import normalize_function
from torch.multiprocessing.reductions import StorageWeakRef
from torch.overrides import TorchFunctionMode
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import (
from torch.utils._pytree import PyTree, tree_map
from torch.utils._stats import count, count_label
from torch.utils.weak import WeakIdRef
class FakeTensorMode(TorchDispatchMode):

    def __init__(self, *, allow_fallback_kernels=True, allow_non_fake_inputs=False, shape_env=None, static_shapes=None):
        log.debug('create_mode 0x%x', id(self))
        self.allow_fallback_kernels = allow_fallback_kernels
        self.fake_tensor_converter = FakeTensorConverter()
        if static_shapes is not None:
            self.static_shapes = static_shapes
        else:
            self.static_shapes = shape_env is None
        import torch._functorch.config
        self.allow_meta = torch._functorch.config.fake_tensor_allow_meta
        self.allow_non_fake_inputs = allow_non_fake_inputs
        self.in_kernel_invocation = False
        self.enter_stack: List[Tuple[bool, Optional[FakeTensorMode]]] = []
        self.shape_env = shape_env
        self.stack = ''.join(traceback.format_stack())
        self._mode_key = torch._C._TorchDispatchModeKey.FAKE

    def is_our_fake(self, t):
        return isinstance(t, FakeTensor) and t.fake_mode is self

    @count
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        assert torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE) is None, func
        try:
            return self.dispatch(func, types, args, kwargs)
        except TypeError:
            log.exception('fake tensor raised TypeError')
            raise

    def __enter__(self):
        maybe_prev_fake_mode = torch._C._unset_dispatch_mode(self._mode_key)
        if self is not maybe_prev_fake_mode:
            self.enter_stack.append((True, maybe_prev_fake_mode))
            return super().__enter__()
        else:
            torch._C._set_dispatch_mode(self)
            self.enter_stack.append((False, None))
        return self

    def __exit__(self, a, b, c):
        live, maybe_prev_fake_mode = self.enter_stack.pop()
        if live:
            out = super().__exit__(a, b, c)
            if maybe_prev_fake_mode is not None:
                torch._C._set_dispatch_mode(maybe_prev_fake_mode)

    def dispatch(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}
        log.debug('%s %s %s', func, args, kwargs)
        if func == torch.ops.prim.device.default:
            assert len(args) == 1 and isinstance(args[0], FakeTensor)
            if args[0].fake_mode.in_kernel_invocation:
                return torch.device('meta')
            else:
                return args[0].fake_device
        elif func is torch.ops.aten.size.default:
            return tuple((int(s) for s in args[0].size()))
        elif func is torch.ops.aten.stride.default:
            return tuple((int(s) for s in args[0].stride()))
        elif func is torch.ops.aten.storage_offset.default:
            return int(args[0].storage_offset())
        if log.getEffectiveLevel() <= logging.DEBUG:
            log.debug('%sFakeTensorMode.__torch_dispatch__: %s', ' ' * RECURSION_COUNT, func)
            incr = IncrementRecursionCount()
        if func in {torch.ops.aten.is_coalesced.default, torch.ops.aten.dense_dim.default, torch.ops.aten.sparse_dim.default}:
            with in_kernel_invocation_manager(self):
                return func(*args, **kwargs)
        flat_args, args_spec = pytree.tree_flatten((args, kwargs))
        flat_arg_fake_tensors = [t for t in flat_args if isinstance(t, FakeTensor) and self.is_our_fake(t)]
        has_symbolic_sizes = any((i._has_symbolic_sizes_strides for i in flat_arg_fake_tensors)) or any((isinstance(a, torch.SymInt) for a in flat_args))
        converter = self.fake_tensor_converter

        def maybe_to_constant(t):
            if isinstance(t, FakeTensor) and self.is_our_fake(t):
                return t.constant
            else:
                return t
        if func in self.lift_fns and (not flat_arg_fake_tensors) or (should_allow_numbers_as_tensors(func) and (not has_symbolic_sizes) and (not flat_arg_fake_tensors)):
            assert all((t.constant is not None for t in flat_arg_fake_tensors)), f'{func} should not have fake inputs without constants'
            const_flat_args = [maybe_to_constant(a) for a in flat_args]
            const_args, const_kwargs = pytree.tree_unflatten(const_flat_args, args_spec)
            out = func(*const_args, **const_kwargs)
            if type(out) is torch.Tensor and self.may_turn_const(out):
                with no_dispatch():
                    out = out.clone()
                return converter(self, out, make_constant=True)
        unrecognized_types = self.check_for_subclass(flat_args)
        if unrecognized_types:
            not_implemented_log.debug('FakeTensorMode unrecognized subclass(es): %s', unrecognized_types)
            return NotImplemented
        if func in self.lift_fns:
            assert len(kwargs) == 0 and len(args) == 1, f'{args} {kwargs}'
            if type(args[0]) is torch.Tensor:
                return converter(self, args[0])
        flat_args, flat_arg_fake_tensors = self.validate_and_convert_non_fake_tensors(func, converter, flat_args, args_spec)
        del args, kwargs
        all_constant = all((e.constant is not None for e in flat_arg_fake_tensors))
        if torch.Tag.nondeterministic_seeded not in func.tags and torch.Tag.inplace_view not in func.tags and all_constant and (len(flat_arg_fake_tensors) != 0) and (not has_symbolic_sizes):
            const_flat_args = [maybe_to_constant(a) for a in flat_args]
            const_args, const_kwargs = pytree.tree_unflatten(const_flat_args, args_spec)
            with no_dispatch():
                out = func(*const_args, **const_kwargs)
            flat_out = pytree.tree_leaves(out)
            flat_out_tensors = [t for t in flat_out if isinstance(t, torch.Tensor)]
            all_constant = all((self.may_turn_const(t) for t in flat_out_tensors))
            if all_constant:
                return pytree.tree_map_only(torch.Tensor, lambda t: converter(self, t, make_constant=True), out)
            for ten in flat_out_tensors:
                converter.invalidate_constant_aliases(ten)
        args, kwargs = pytree.tree_unflatten(flat_args, args_spec)
        self.invalidate_written_to_constants(func, flat_arg_fake_tensors, args, kwargs)
        if has_symbolic_sizes:
            fast_impl = get_fast_op_impls().get(func)
            if fast_impl is not None:
                return fast_impl(self, *args, **kwargs)
        from torch._decomp import meta_table as meta_table
        if func not in meta_table and (not self.cpp_meta_supports_symint(func)):
            from torch._decomp import decomposition_table
            if func in decomposition_table and (has_symbolic_sizes or (torch_decomp_decompositions(func) and all((not e.is_sparse for e in flat_arg_fake_tensors)))):
                with self:
                    return decomposition_table[func](*args, **kwargs)
            with self:
                r = func.decompose(*args, **kwargs)
                if r is not NotImplemented:
                    return r
        if 'prims::' in func._schema.name and hasattr(func, 'prim_meta_impl') and (not stride_incorrect_op(func)):
            with self:
                return func.prim_meta_impl(*args, **kwargs)
        maybe_abstract_impl = torch._library.simple_registry.singleton.find(func.name()).abstract_impl.kernel
        if maybe_abstract_impl:
            ctx = torch._library.abstract_impl.AbstractImplCtx(self.shape_env, func)
            with torch._library.abstract_impl.set_ctx_getter(lambda: ctx), self:
                result = maybe_abstract_impl(*args, **kwargs)
                return result
        for run_impl_check, op_impl in op_implementations:
            if func in (aten._nested_tensor_from_tensor_list.default, aten._nested_tensor_from_tensor_list.out):
                raise UnsupportedOperatorException('torch.compile does not support strided NestedTensor')
            if run_impl_check(func):
                op_impl_out = op_impl(self, func, *args, **kwargs)
                if op_impl_out != NotImplemented:
                    return op_impl_out

        def can_run_unsafe_fallback(func: OpOverload):
            if not self.allow_fallback_kernels:
                return False
            allowed_namespaces = {'debugprims', 'prims', 'aten', 'xla', 'vision', 'torchtext', 'torchaudio', 'quantized'}
            grandfathered_ops_FIXME = {'fbgemm::gmm'}
            return func.namespace in allowed_namespaces or func.name() in grandfathered_ops_FIXME

        def maybe_run_unsafe_fallback(error=None):
            from torch._higher_order_ops.auto_functionalize import can_auto_functionalize
            if can_auto_functionalize(func):
                return None
            if has_symbolic_sizes or not can_run_unsafe_fallback(func):
                raise UnsupportedOperatorException(func)
            if error is None:
                error = UnsupportedOperatorException(func)
            return run_fallback_kernel(self, func, flat_args, args_spec, error)
        if not torch._C._dispatch_has_computed_kernel_for_dispatch_key(func.name(), 'Meta'):
            return maybe_run_unsafe_fallback()
        try:
            with in_kernel_invocation_manager(self):
                r = func(*args, **kwargs)
        except NotImplementedError as not_implemented_error:
            return maybe_run_unsafe_fallback(not_implemented_error)
        return self.wrap_meta_outputs_with_default_device_logic(r, func, flat_args, device=kwargs.get('device'))

    def check_for_subclass(self, flat_args):

        def check(x):
            return isinstance(x, torch.Tensor) and (not isinstance(x, FakeTensor)) and (type(x) is not torch.Tensor) and (type(x) is not torch.nn.Parameter)
        return [type(x) for x in flat_args if check(x)]

    def validate_and_convert_non_fake_tensors(self, func, converter, flat_args, args_spec):
        """
        Checks if the list of tensors are fake tensors.
        If not, try to convert them to fake tensors.
        Returns the original args, kwargs, and a flattened list of (args, kwargs) that are fake tensors.
        """
        flat_arg_fake_tensors = []

        def validate(x):
            if not isinstance(x, torch.Tensor):
                return x
            nonlocal flat_arg_fake_tensors
            if not self.is_our_fake(x):
                if torch.Tag.inplace_view in func.tags:
                    args, kwargs = pytree.tree_unflatten(flat_args, args_spec)
                    raise Exception(f"Can't call metadata mutating ops on non-Fake Tensor inputs. Found in {render_call(func, args, kwargs)}")
                if not self.allow_non_fake_inputs:
                    if isinstance(x, FakeTensor) and x.fake_mode is not self:
                        raise AssertionError('Mixing fake modes NYI')
                    args, kwargs = pytree.tree_unflatten(flat_args, args_spec)
                    raise Exception(f"Please convert all Tensors to FakeTensors first or instantiate FakeTensorMode with 'allow_non_fake_inputs'. Found in {render_call(func, args, kwargs)}")
                x = converter(self, x)
            flat_arg_fake_tensors.append(x)
            return x
        validated_args = [validate(a) for a in flat_args]
        return (validated_args, flat_arg_fake_tensors)

    def wrap_meta_outputs_with_default_device_logic(self, r, func, flat_args, device):
        converter = self.fake_tensor_converter
        common_device = None
        has_scalar_only_inputs = False

        def wrap(e):
            nonlocal common_device
            nonlocal has_scalar_only_inputs
            if isinstance(e, torch.Tensor) and common_device is None:
                common_device, has_scalar_only_inputs = FakeTensor._find_common_device(func, flat_args)
            if self.is_our_fake(e):
                torch._check(e.device == common_device, lambda: f'FakeTensor is wrapped to wrong device, found {e.device}, expected {common_device}')
            if isinstance(e, torch.Tensor) and (not self.is_our_fake(e)) and (converter is not None):
                if has_scalar_only_inputs:
                    return converter(self, e)
                else:
                    return converter.from_meta_and_device(self, e, device or common_device)
            else:
                return e
        return tree_map(wrap, r)

    def cpp_meta_supports_symint(self, func):
        if torch.Tag.view_copy in func.tags:
            return True
        return func in [aten.empty.memory_format, aten.empty_strided.default, aten.as_strided_scatter.default, aten.as_strided.default, aten.as_strided_.default, aten.zeros.default, aten.detach.default, aten.view_as_real.default, aten.view_as_complex.default, aten.set_.source_Storage_storage_offset, aten._sparse_coo_tensor_with_dims_and_tensors.default]

    @property
    def lift_fns(self):
        return (aten.lift_fresh.default, aten.lift_fresh_copy.default)

    def may_turn_const(self, t):
        return t.numel() <= CONSTANT_NUMEL_LIMIT and (not t.is_sparse) and (not self.is_our_fake(t)) and (not t.device.type == 'meta')

    def invalidate_written_to_constants(self, func, flat_arg_fake_tensors, args, kwargs):
        any_constant = any((e.constant is not None for e in flat_arg_fake_tensors))
        schema_info = get_schema_info(func)
        if any_constant and schema_info.is_mutable():
            _, new_kwargs = normalize_function(func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True)
            for k, v in new_kwargs.items():
                k = k if k != 'input' or schema_info.has_argument(k) else 'self'
                if self.is_our_fake(v) and schema_info.is_mutable(k) and (v.constant is not None):
                    self.fake_tensor_converter.invalidate_constant_aliases(v.constant)

    def from_tensor(self, tensor, *, static_shapes=None, source: Optional[Source]=None, symbolic_context=None, memoized_only=False):
        shape_env = self.shape_env
        if static_shapes is None:
            static_shapes = self.static_shapes
        if static_shapes:
            assert symbolic_context is None, 'cannot set both static_shapes and symbolic_context'
            shape_env = None
        if not symbolic_context and (not source) and (not static_shapes):
            if (tracing_context := torch._guards.TracingContext.try_get()):
                if tensor in tracing_context.tensor_to_context:
                    symbolic_context = tracing_context.tensor_to_context[tensor]
                    source = symbolic_context.tensor_source
        return self.fake_tensor_converter(self, tensor, shape_env=shape_env, source=source, symbolic_context=symbolic_context, memoized_only=memoized_only)