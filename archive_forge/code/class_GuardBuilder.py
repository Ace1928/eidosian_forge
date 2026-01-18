from __future__ import annotations
import ast
import builtins
import collections
import dataclasses
import enum
import functools
import importlib
import inspect
import itertools
import logging
import math
import os
import re
import sys
import textwrap
import types
import weakref
from inspect import currentframe, getframeinfo
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from weakref import ReferenceType
import torch
import torch.utils._device
from torch._dynamo.source import (
from torch._guards import (
from torch.fx.experimental.symbolic_shapes import (
from torch.utils._traceback import format_frame, report_compile_source_on_error
from torch.utils.weak import TensorWeakRef
from . import config, convert_frame, exc, mutation_guard
from .eval_frame import set_guard_error_hook
from .source import DefaultsSource, LocalSource, TypeSource
from .types import GuardedCode, GuardFail, GuardFn  # noqa: F401
from .utils import (
class GuardBuilder(GuardBuilderBase):

    def __init__(self, id_ref: Callable[[Any], str], source_ref: Callable[[Source], str], lookup_weakrefs: Callable[[object], ReferenceType[object]], local_scope: Dict[str, object], global_scope: Dict[str, object], check_fn_manager: CheckFunctionManager):
        self.id_ref = id_ref
        self.source_ref = source_ref
        self.lookup_weakrefs = lookup_weakrefs
        self.scope: Dict[str, Dict[str, object]] = {'L': local_scope, 'G': global_scope}
        self.scope['__builtins__'] = builtins.__dict__.copy()
        for name, package_module in torch.package.package_importer._package_imported_modules.items():
            name = name.replace('>', '_').replace('<', '_').replace('.', '_dot_')
            self.scope['__builtins__'][name] = package_module
            self.scope[name] = package_module
        self.argnames: List[str] = []
        self.code: List[GuardCodeList] = []
        self.shape_env_code: List[GuardCodeList] = []
        self.tensor_check_names: List[str] = []
        self.tensor_check_examples: List[torch.Tensor] = []
        self.tensor_check_guards: List[Guard] = []
        self.check_fn_manager: CheckFunctionManager = check_fn_manager
        self.id_matched_objs: Dict[str, ReferenceType[object]] = {}
        self.config_hash: Optional[bytes] = None

    def get(self, name: str) -> Any:
        return eval(name, self.scope, CLOSURE_VARS)

    def arg_ref(self, guard: Union[str, Guard]) -> str:
        name: str
        if isinstance(guard, str):
            name = guard
        else:
            name = guard.name
        base = strip_getattr_getitem(strip_function_call(name))
        if base not in self.argnames:
            if re.match('[a-zA-Z0-9_]+', base):
                if re.match('^\\d+$', base):
                    log.warning('invalid var name: %s', guard)
                self.argnames.append(base)
        return name

    def TYPE_MATCH(self, guard: Guard) -> None:
        t = type(self.get(guard.name))
        obj_id = self.id_ref(t)
        code = f'___check_type_id({self.arg_ref(guard)}, {obj_id})'
        self._produce_guard_code(guard, [code])

    def DICT_VERSION(self, guard: Guard):
        ref = self.arg_ref(guard)
        version = dict_version(self.get(guard.name))
        code = f'___dict_version({ref}) == {version}'
        self._produce_guard_code(guard, [code])

    def DICT_CONTAINS(self, guard: Guard, key: str, invert: bool):
        dict_ref = self.arg_ref(guard)
        maybe_not = 'not ' if invert else ''
        code = f'{maybe_not}___dict_contains({key!r}, {dict_ref})'
        return self._produce_guard_code(guard, [code])

    def BOOL_FALSE(self, guard: Guard):
        ref = self.arg_ref(guard)
        code = f'not {ref}'
        self._produce_guard_code(guard, [code])

    def ID_MATCH(self, guard: Guard):
        if isinstance(guard.originating_source, TypeSource):
            return self.TYPE_MATCH(Guard(guard.originating_source.base, GuardBuilder.TYPE_MATCH))
        ref = self.arg_ref(guard)
        val = self.get(guard.name)
        code = f'___check_obj_id({ref}, {self.id_ref(val)})'
        self._produce_guard_code(guard, [code])
        if isinstance(guard.originating_source, LocalSource):
            if isinstance(val, torch.nn.Module):
                local_name = guard.originating_source.local_name
                weak_id = self.lookup_weakrefs(val)
                if weak_id is not None:
                    self.id_matched_objs[local_name] = weak_id

    def NAME_MATCH(self, guard: Guard):
        obj = self.get(guard.name)
        code = f"{self.arg_ref(guard)}.__name__ == '{obj.__name__}'"
        self._produce_guard_code(guard, [code])

    def DATA_PTR_MATCH(self, guard: Guard):
        obj = self.get(guard.name)
        code = f'{self.arg_ref(guard)}.data_ptr() == {obj.data_ptr()}'
        self._produce_guard_code(guard, [code])

    def HASATTR(self, guard: Guard):
        m = re.match('^(.*)[.]([a-zA-Z0-9_]+)$', guard.name)
        assert m, f'invalid hasattr check {guard.name}'
        base, attr = m.group(1, 2)
        ref = self.arg_ref(base)
        val = hasattr(self.get(base), attr)
        code = None
        if val:
            code = f'hasattr({ref}, {attr!r})'
        else:
            code = f'not hasattr({ref}, {attr!r})'
        self._produce_guard_code(guard, [code], provided_guarded_object=self.get(base))

    def EQUALS_MATCH(self, guard: Guard):
        ref = self.arg_ref(guard)
        val = self.get(guard.name)
        t = type(val)
        if np:
            np_types: Tuple[Type[Any], ...] = (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64)
        else:
            np_types = ()
        ok_types = (int, float, bool, type(None), str, type, list, tuple, set, slice, frozenset, range, torch.Size, torch.device, torch.dtype, *np_types)
        if istype(val, dict):
            assert all((istype(x, ok_types) for x in itertools.chain(val.keys(), val.values())))
        else:
            assert istype(val, ok_types), t.__name__
        if istype(val, float) and math.isnan(val):
            code = list()
            code.append(f'___check_type_id({ref}, {self.id_ref(t)})')
            code.append(f'__math_isnan({ref})')
            self._produce_guard_code(guard, code)
            return
        code = list()
        if istype(val, (list, tuple)):
            self.LIST_LENGTH(guard)
            for idx, elem in enumerate(val):
                code.append(f'___check_type_id({ref}[{idx}], {self.id_ref(type(elem))})')
        else:
            code.append(f'___check_type_id({ref}, {self.id_ref(t)})')
        if istype(val, torch.Size):
            val = tuple(val)
        code.append(f'{ref} == {val!r}')
        self._produce_guard_code(guard, code)

    def CONSTANT_MATCH(self, guard: Guard):
        val = self.get(guard.name)
        if istype(val, (bool, type(None))):
            self.ID_MATCH(guard)
        else:
            self.EQUALS_MATCH(guard)

    def NN_MODULE(self, guard: Guard):
        self.ID_MATCH(guard)
        ref = self.arg_ref(guard)
        val = self.get(guard.name)

        def setup_guard():
            assert istype(val.training, bool)
            self.code.append(GuardCodeList([f'{ref}.training == {val.training}'], guard))
        if hasattr(val, 'training'):
            setup_guard()
        else:
            exc.unimplemented(f'Guard setup for uninitialized class {type(val)}')

    def FUNCTION_MATCH(self, guard: Guard):
        """things like torch.add and user defined functions"""
        if guard.is_local():
            return self.ID_MATCH(guard)

    def CLOSURE_MATCH(self, guard: Guard):
        """matches a closure by __code__ id."""
        if guard.is_local():
            val = self.get(guard.name)
            if type(val) == types.FunctionType and hasattr(val, '__code__'):
                ref = self.arg_ref(guard)
                code = [f"___check_obj_id(getattr({ref}, '__code__', None), {self.id_ref(val.__code__)})"]
                self._produce_guard_code(guard, code)
            else:
                self.FUNCTION_MATCH(guard)

    def BUILTIN_MATCH(self, guard: Guard):
        return self.FUNCTION_MATCH(guard)

    def PYMODULE_MATCH(self, guard: Guard):
        return self.FUNCTION_MATCH(guard)

    def LIST_LENGTH(self, guard):
        ref = self.arg_ref(guard)
        value = self.get(guard.name)
        t = type(value)
        code = list()
        code.append(f'___check_type_id({ref}, {self.id_ref(t)})')
        code.append(f'len({ref}) == {len(value)}')
        self._produce_guard_code(guard, code)

    def TUPLE_ITERATOR_LEN(self, guard):
        ref = self.arg_ref(guard)
        value = self.get(guard.name)
        t = type(value)
        code = list()
        code.append(f'___check_type_id({ref}, {self.id_ref(t)})')
        code.append(f'___tuple_iterator_len({ref}) == {tuple_iterator_len(value)}')
        self._produce_guard_code(guard, code)

    def DUPLICATE_INPUT(self, guard, source_b):
        ref_a = self.arg_ref(guard)
        ref_b = self.arg_ref(source_b.name())
        code = [f'{ref_b} is {ref_a}']
        self._produce_guard_code(guard, code)

    def DICT_KEYS(self, guard):
        ref = self.arg_ref(guard)
        value = self.get(guard.name)
        t = type(value)
        code = list()
        code.append(f'___check_type_id({ref}, {self.id_ref(t)})')
        param_key_ids = set(dict_param_key_ids(value))
        const_keys = set(dict_const_keys(value))
        const_keys_repr = dict_const_keys_repr(const_keys, local=is_from_local_source(guard.originating_source))
        if param_key_ids:
            code.append(f'___dict_param_key_ids({ref}) == {param_key_ids!r}')
            code.append(f'___dict_const_keys({ref}) == {const_keys_repr}')
        else:
            code.append(f'set({ref}.keys()) == {const_keys_repr}')
        self._produce_guard_code(guard, code)

    def WEAKREF_ALIVE(self, guard):
        self._produce_guard_code(guard, [f'{self.arg_ref(guard)} is not None'])

    def NN_MODULE_PARAM_NAMES(self, guard):
        ref = self.arg_ref(guard)
        value = self.get(guard.name)
        t = type(value)
        keys = {k for k, v in value.named_parameters()}
        code = list()
        code.append(f'___check_type_id({ref}, {self.id_ref(t)})')
        code.append(f'{{k for k, v in {ref}.named_parameters()}} == {keys!r}')
        self._produce_guard_code(guard, code)

    def ODICT_KEYS(self, guard):
        """OrderedDict keys match"""
        ref = self.arg_ref(guard)
        value = self.get(guard.name)
        t = type(value)
        code = list()
        code.append(f'___check_type_id({ref}, {self.id_ref(t)})')
        code.append(f'str({ref}.keys()) == {str(value.keys())!r}')
        self._produce_guard_code(guard, code)

    def OBJECT_MUTATION(self, guard: Guard):
        mutation_guard.watch(self.get(guard.name), self.check_fn_manager)

    def GRAD_MODE(self, guard: Guard):
        pass

    def DETERMINISTIC_ALGORITHMS(self, guard: Guard):
        pass

    def TORCH_FUNCTION_STATE(self, guard: Guard):
        pass

    def DEFAULT_DEVICE(self, guard: Guard):
        """Guard on CURRENT_DEVICE per torch.utils._device"""
        assert guard.source is GuardSource.GLOBAL
        import torch.utils._device as m
        self._produce_guard_code(guard, [f'utils_device.CURRENT_DEVICE == {m.CURRENT_DEVICE!r}'])

    def BACKEND_MATCH(self, guard: Guard):
        """Guard on backend matching based on id of current_backend"""
        assert guard.source is GuardSource.GLOBAL
        backend_id = f'{id(torch._dynamo.eval_frame.guarded_backend_cache.current_backend)}'
        code = [f'(___skip_backend_check() or ___current_backend() == ___lookup_backend({backend_id}))']
        self._produce_guard_code(guard, code)

    def CONFIG_HASH_MATCH(self, guard: Guard):
        """Guard on the hash of the compiled function's dynamo config"""
        config_hash = torch._dynamo.eval_frame.get_saved_else_current_config_hash()
        assert guard.source is GuardSource.GLOBAL
        code = [f"___compile_config_hash() == '{config_hash.hex()}'"]
        self.config_hash = config_hash
        self._produce_guard_code(guard, code)

    def HAS_GRAPH_BREAK(self, guard: Guard):
        code = ['not ___needs_nopython()']
        self._produce_guard_code(guard, code)

    def SHAPE_ENV(self, guard: Guard):
        assert guard.name == ''
        output_graph = self.check_fn_manager.output_graph
        fs = output_graph.tracked_fakes
        constraint_inputs = [a.constraint_dims for a in fs]

        def get_sources(t_id, dim):
            return [TensorPropertySource(source, TensorProperty.SIZE, dim) for source in output_graph.tracked_fakes_id_to_source[t_id]]
        if output_graph.export_constraints:
            source_pairs: List[Tuple[Source, Source]] = []
            for constraint in output_graph.export_constraints:
                if constraint.t_id in output_graph.tracked_fakes_id_to_source:
                    source, *other_sources = get_sources(constraint.t_id, constraint.dim)
                    source_pairs.extend(((source, other_source) for other_source in other_sources))
                    if constraint.shared is not None:
                        other_sources = get_sources(constraint.shared.t_id, constraint.shared.dim)
                        source_pairs.extend(((source, other_source) for other_source in other_sources))
                else:
                    log.warning('Untracked tensor used in export constraints')
            equalities_inputs = EqualityConstraint(source_pairs=source_pairs, warn_only=False)
        else:
            equalities_inputs = None
        guards = output_graph.shape_env.produce_guards([a.fake for a in fs], [a.source for a in fs], constraint_inputs=constraint_inputs, equalities_inputs=equalities_inputs, source_ref=self.source_ref, ignore_static=not self.check_fn_manager.output_graph.export)
        output_graph.shape_env.freeze()
        for shape_guard in guards:
            self._produce_guard_code(guard, [shape_guard], shape_env=True)

    def TENSOR_MATCH(self, guard: Guard, value=None):
        if guard.is_nn_module():
            self.ID_MATCH(guard)
        else:
            if isinstance(value, TensorWeakRef):
                value = value()
            value = value if value is not None else self.get(guard.name)
            assert isinstance(value, torch.Tensor)
            tensor_name = self.arg_ref(guard)
            code: List[str] = list()
            if self.check_fn_manager.output_graph.export:
                self.TYPE_MATCH(guard)
                terms = ['dtype', 'device', 'requires_grad', 'ndimension()']
                for term in terms:
                    real_value = self.get(tensor_name + '.' + term)
                    if istype(real_value, (torch.device, torch.dtype)):
                        code.append(f'str({tensor_name}.{term}) == {str(real_value)!r}')
                    else:
                        code.append(f'{tensor_name}.{term} == {real_value}')
            else:
                self.tensor_check_names.append(tensor_name)
                self.tensor_check_examples.append(value)
                self.tensor_check_guards.append(guard)
            assert guard.source is not None
            static, reason = tensor_always_has_static_shape(value, is_tensor=True, guard_source=guard.source)
            if not static:
                if hasattr(value, '_dynamo_dynamic_indices'):
                    code.append(f"(({tensor_name}._dynamo_dynamic_indices.issubset({value._dynamo_dynamic_indices})) if hasattr({tensor_name}, '_dynamo_dynamic_indices') else True)")
                else:
                    code.append(f"hasattr({tensor_name}, '_dynamo_dynamic_indices') == False")
            if len(code) > 0:
                self._produce_guard_code(guard, code)

    def _produce_guard_code(self, guard, code_list, provided_guarded_object=None, shape_env=False):
        cur_frame = currentframe()
        assert cur_frame is not None
        caller = cur_frame.f_back
        del cur_frame
        assert caller is not None
        func_name = getframeinfo(caller)[2]
        del caller
        assert func_name in dir(self.__class__), f'_produce_guard_code must be called from inside GuardedCode. Called from {func_name}'
        if shape_env:
            self.shape_env_code.append(GuardCodeList(code_list, guard))
        else:
            self.code.append(GuardCodeList(code_list, guard))
        if provided_guarded_object is None:
            name_valid = guard.name is not None and guard.name != ''
            guarded_object = self.get(guard.name) if name_valid else None
        else:
            guarded_object = provided_guarded_object
        guarded_object_type = weakref.ref(type(guarded_object)) if guarded_object is not None else None
        obj_ref = None
        if hasattr(guarded_object.__class__, '__weakref__') and (not isinstance(guarded_object, enum.Enum)):
            obj_ref = weakref.ref(guarded_object)
        guard.set_export_info(func_name, guarded_object_type, code_list, obj_ref)