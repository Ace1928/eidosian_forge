import functools
import warnings
from typing import Callable, Union
import torch
import torch.utils._pytree as pytree
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import (
from torch.utils._python_dispatch import TorchDispatchMode
class CrossRefFakeMode(TorchDispatchMode):

    def __init__(self, ignore_op_fn: Union[Callable[[OpOverload], bool], None]=None, *, check_strides=True, check_aliasing=True):
        self.ignore_op_fn = ignore_op_fn if ignore_op_fn is not None else lambda fn: False
        self.check_strides = check_strides
        self.check_aliasing = check_aliasing

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        fake_r = None
        if func not in (aten.lift_fresh.default, aten.lift_fresh_copy.default, aten.set_.source_Storage_storage_offset) and (not self.ignore_op_fn(func)) and (torch.Tag.dynamic_output_shape not in func.tags) and (torch.Tag.inplace_view not in func.tags) and (torch.Tag.data_dependent_output not in func.tags):
            from torch.fx.experimental.symbolic_shapes import ShapeEnv
            try:
                with FakeTensorMode(shape_env=ShapeEnv()) as fake_mode:
                    fake_args, fake_kwargs = pytree.tree_map_only(torch.Tensor, functools.partial(fake_mode.from_tensor, static_shapes=True), (args, kwargs))
                    with warnings.catch_warnings():
                        fake_r = func(*fake_args, **fake_kwargs)
            except UnsupportedFakeTensorException:
                pass
        context = f'When comparing the output of {func} on FakeTensor and concrete Tensors, found'
        r = func(*args, **kwargs)
        if fake_r is not None:
            r_flat = pytree.tree_leaves(r)
            f_flat = pytree.tree_leaves(fake_r)
            assert len(f_flat) == len(r_flat), f'{context} mismatch in number of returns {len(f_flat)} != {len(r_flat)}'
            if self.check_aliasing:
                r_aliasing = outputs_alias_inputs(r, (args, kwargs))
                f_aliasing = outputs_alias_inputs(fake_r, (fake_args, fake_kwargs))
                assert r_aliasing == f_aliasing, f'{context} mismatch in outputs_alias_inputs check {f_aliasing} != {r_aliasing}'
                r_identity_eq = outputs_are_inputs(r, (args, kwargs))
                f_identity_eq = outputs_are_inputs(fake_r, (fake_args, fake_kwargs))
                assert r_identity_eq == f_identity_eq, f'{context} mismatch in outputs_are_inputs check {f_identity_eq} != {r_identity_eq}'
                r_output_alias_each_other = output_alias_each_other(r)
                f_output_alias_each_other = output_alias_each_other(fake_r)
                assert r_output_alias_each_other == f_output_alias_each_other, f'{context} mismatch in outputs_alias_each_other check {f_output_alias_each_other} != {r_output_alias_each_other}'
            for idx, (r_out, fake_out) in enumerate(zip(pytree.tree_leaves(r), pytree.tree_leaves(fake_r))):
                r_is_ten = isinstance(r_out, torch.Tensor)
                assert r_is_ten == isinstance(fake_out, torch.Tensor), f'{context} mismatched number of tensor outputs'
                if r_is_ten:
                    assert r_out.requires_grad == fake_out.requires_grad, f'{context} mismatched requires_grad-ness of outputs. This usually means that you have added autograd support for your operator at a dispatch key other than Autograd, which will lead to problems'
                    if torch._C._has_storage(r_out):
                        r_offset = r_out.storage_offset()
                        f_offset = fake_out.storage_offset()
                        assert r_offset == f_offset, f'{context} mismatched storage offset'
                    try:
                        torch._prims.utils.compare_tensor_meta(r_out, fake_out, check_strides=self.check_strides, allow_rhs_unbacked=True)
                    except Exception as e:
                        if is_sdpa_error(func, idx, e):
                            continue
                        error_message = f'{context} mismatched tensor metadata: {e}' if len(r_flat) == 1 else f'{context} mismatched tensor metadata for output[{idx}]: {e}'
                        raise RuntimeError(error_message) from e
        return r