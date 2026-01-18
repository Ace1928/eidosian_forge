import dataclasses
import itertools
import re
from dataclasses import dataclass
from enum import auto, Enum
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.utils import assert_never, NamespaceHelper, OrderedSet
@dataclass(frozen=True)
class NativeFunction:
    namespace: str
    func: 'FunctionSchema'
    use_const_ref_for_mutable_tensors: bool
    device_guard: bool
    device_check: DeviceCheckType
    python_module: Optional[str]
    category_override: Optional[str]
    variants: Set[Variant]
    manual_kernel_registration: bool
    manual_cpp_binding: bool
    loc: 'Location'
    autogen: List['OperatorName']
    ufunc_inner_loop: Dict[UfuncKey, 'UfuncInnerLoop']
    structured: bool
    structured_delegate: Optional['OperatorName']
    structured_inherits: Optional[str]
    precomputed: Optional['Precompute']
    cpp_no_default_args: Set[str]
    is_abstract: bool
    has_composite_implicit_autograd_kernel: bool
    has_composite_implicit_autograd_nested_tensor_kernel: bool
    has_composite_explicit_autograd_kernel: bool
    has_composite_explicit_autograd_non_functional_kernel: bool
    tags: Set[str]

    @staticmethod
    def from_yaml(ei: Dict[str, object], loc: 'Location', valid_tags: Set[str], ignore_keys: Optional[Set[DispatchKey]]=None) -> Tuple['NativeFunction', Dict[DispatchKey, Dict['OperatorName', 'BackendMetadata']]]:
        """
        Parse a NativeFunction from a dictionary as directly parsed
        from native_functions.yaml
        """
        e = ei.copy()
        funcs = e.pop('func')
        assert isinstance(funcs, str), f'not a str: {funcs}'
        namespace_helper = NamespaceHelper.from_namespaced_entity(namespaced_entity=funcs, max_level=1)
        namespace = namespace_helper.get_cpp_namespace(default='aten')
        func = FunctionSchema.parse(namespace_helper.entity_name)
        cpp_no_default_args_list = e.pop('cpp_no_default_args', [])
        assert isinstance(cpp_no_default_args_list, list)
        cpp_no_default_args = set(cpp_no_default_args_list)
        use_const_ref_for_mutable_tensors = e.pop('use_const_ref_for_mutable_tensors', False)
        assert isinstance(use_const_ref_for_mutable_tensors, bool)
        variants_s = e.pop('variants', 'function')
        assert isinstance(variants_s, str)
        variants: Set[Variant] = set()
        for v in variants_s.split(', '):
            if v == 'function':
                variants.add(Variant.function)
            elif v == 'method':
                variants.add(Variant.method)
            else:
                raise AssertionError(f'illegal variant {v}')
        manual_kernel_registration = e.pop('manual_kernel_registration', False)
        assert isinstance(manual_kernel_registration, bool), f'not a bool: {manual_kernel_registration}'
        manual_cpp_binding = e.pop('manual_cpp_binding', False)
        assert isinstance(manual_cpp_binding, bool), f'not a bool: {manual_cpp_binding}'
        device_guard = e.pop('device_guard', True)
        assert isinstance(device_guard, bool), f'not a bool: {device_guard}'
        device_check_s = e.pop('device_check', None)
        assert device_check_s is None or isinstance(device_check_s, str), f'not a str: {device_check_s}'
        device_check: DeviceCheckType
        if device_check_s is None:
            device_check = DeviceCheckType.ExactSame
        else:
            device_check = DeviceCheckType[device_check_s]
        structured = e.pop('structured', False)
        assert isinstance(structured, bool), f'not a bool: {structured}'
        structured_delegate_s = e.pop('structured_delegate', None)
        assert structured_delegate_s is None or isinstance(structured_delegate_s, str), f'not a str: {structured_delegate_s}'
        assert structured_delegate_s is None or '::' not in structured_delegate_s, 'namespace is not supported in structured delegate, using the same namespace as the native function'
        structured_delegate: Optional[OperatorName] = None
        if structured_delegate_s is not None:
            structured_delegate = OperatorName.parse(structured_delegate_s)
        structured_inherits = e.pop('structured_inherits', None)
        assert structured_inherits is None or isinstance(structured_inherits, str), f'not a str: {structured_inherits}'
        assert structured_inherits is None or '::' not in structured_inherits, 'namespace is not supported in structured inherits, using the same namespace as the native function'
        python_module = e.pop('python_module', None)
        assert python_module is None or isinstance(python_module, str), f'not a str: {python_module}'
        assert python_module is None or Variant.method not in variants, 'functions in modules cannot be methods'
        category_override = e.pop('category_override', None)
        assert category_override is None or isinstance(category_override, str), f'not a str: {category_override}'
        precomputed_dict = e.pop('precomputed', None)
        assert precomputed_dict is None or structured is True
        precomputed = Precompute.parse(precomputed_dict) if precomputed_dict else None
        tags_inp = e.pop('tags', [])
        if isinstance(tags_inp, str):
            tags_inp = [tags_inp]
        assert isinstance(tags_inp, list)
        if namespace == 'aten' and 'pt2_compliant_tag' in valid_tags:
            tags_inp.append('pt2_compliant_tag')
        tags: Set[str] = set()
        for t in tags_inp:
            assert len(valid_tags) > 0
            if t in valid_tags:
                tags.add(t)
            else:
                raise AssertionError(f'illegal tag {t}')
        from torchgen.api import cpp
        raw_dispatch = e.pop('dispatch', None)
        assert raw_dispatch is None or isinstance(raw_dispatch, dict), e
        dispatch: Dict[DispatchKey, BackendMetadata] = {}
        num_dispatch_keys: int = 0
        if raw_dispatch is not None:
            assert not manual_kernel_registration, 'cannot specify both manual_kernel_registration and dispatch; with manual registration, dispatch has no effect!'
            redundant_composite_implicit_autograd = False
            for ks, v in raw_dispatch.items():
                if ks == '__line__':
                    continue
                assert isinstance(ks, str), e
                for k in ks.split(','):
                    dispatch_key = DispatchKey.parse(k.strip())
                    num_dispatch_keys += 1
                    if ignore_keys and dispatch_key in ignore_keys:
                        continue
                    assert dispatch_key in dispatch_keys, f'Dispatch key {dispatch_key} of kernel {v} is not a supported dispatch key.'
                    namespace_helper = NamespaceHelper.from_namespaced_entity(v, max_level=3)
                    kernel_namespace = namespace_helper.get_cpp_namespace(default='at')
                    dispatch[dispatch_key] = BackendMetadata(kernel=namespace_helper.entity_name, structured=structured and is_structured_dispatch_key(dispatch_key), cpp_namespace=kernel_namespace + '::native')
                    if dispatch_key is DispatchKey.CompositeImplicitAutograd and v == cpp.name(func):
                        redundant_composite_implicit_autograd = True
            assert not (num_dispatch_keys == 1 and redundant_composite_implicit_autograd), 'unnecessary dispatch table for this function; just delete the dispatch key entirely'
            assert structured_delegate or dispatch.keys() != {DispatchKey.CompositeImplicitAutograd} or dispatch[DispatchKey.CompositeImplicitAutograd].supports_symint() or (num_dispatch_keys != 1), f'unexpected name for singleton CompositeImplicitAutograd dispatch entry: expected {cpp.name(func)} but got {dispatch[DispatchKey.CompositeImplicitAutograd]}.  Rename your implementation to the expected name, then delete the dispatch table'
        elif not structured and structured_delegate is None:
            name = str(func.name.name)
            assert not (name.startswith('new_') or name.endswith('_like') or (func.arguments.tensor_options and (not func.arguments.has_tensor_arg()))), f'expected {name} to have a CompositeExplicitAutograd dispatch entry, but there was no dispatch table.  Factory functions should not have implicit dispatch as they should not be decomposed for __torch_dispatch__'
            dispatch[DispatchKey.CompositeImplicitAutograd] = BackendMetadata(cpp.name(func), structured=False, cpp_namespace=DEFAULT_KERNEL_NAMESPACE)
        composites_in_dispatch = [d for d in dispatch if d == DispatchKey.CompositeExplicitAutograd or d == DispatchKey.CompositeExplicitAutogradNonFunctional or d == DispatchKey.CompositeImplicitAutograd or (d == DispatchKey.CompositeImplicitAutogradNestedTensor)]
        assert len(composites_in_dispatch) <= 1 or (len(composites_in_dispatch) == 2 and DispatchKey.CompositeExplicitAutogradNonFunctional not in composites_in_dispatch and (DispatchKey.CompositeImplicitAutogradNestedTensor in composites_in_dispatch)), 'cannot specify more than one of CompositeExplicitAutograd, CompositeExplicitAutogradNonFunctional, or CompositeImplicitAutograd on a single kernel; each strictly subsumes the other.  If you wanted to provide an explicit autograd implementation, specify CompositeExplicitAutograd; otherwise specify CompositeImplicitAutograd only'
        autogen_str = e.pop('autogen', '')
        assert isinstance(autogen_str, str)
        autogen = [] if autogen_str == '' else [OperatorName.parse(x) for x in autogen_str.split(', ')]
        raw_ufunc_inner_loop = e.pop('ufunc_inner_loop', {})
        ufunc_inner_loop = {}
        if isinstance(raw_ufunc_inner_loop, str):
            ufunc_inner_loop[UfuncKey.Generic] = UfuncInnerLoop.parse(raw_ufunc_inner_loop, UfuncKey.Generic)
        elif isinstance(raw_ufunc_inner_loop, dict):
            for k, vo in raw_ufunc_inner_loop.items():
                if k == '__line__':
                    continue
                assert isinstance(k, str), f'ufunc_inner_loop key is not a str: {k}'
                assert isinstance(vo, str), f'ufunc_inner_loop value is not a str: {v}'
                ufunc_key = UfuncKey.parse(k)
                ufunc_inner_loop[ufunc_key] = UfuncInnerLoop.parse(vo, ufunc_key)
        else:
            raise AssertionError(f'ufunc_inner_loop not str or dict: {raw_ufunc_inner_loop}')
        if ufunc_inner_loop:
            assert structured, 'ufunc must be structured'
            import torchgen.api.ufunc as ufunc
            for dispatch_key in UFUNC_DISPATCH_KEYS:
                assert dispatch_key not in dispatch, f'ufunc should not have explicit dispatch entry for {dispatch_key}'
                dispatch[dispatch_key] = BackendMetadata(kernel=ufunc.schema_kernel_name(func, dispatch_key), structured=True, cpp_namespace=DEFAULT_KERNEL_NAMESPACE)
        if structured_delegate:
            is_abstract = True
        else:
            is_abstract = dispatch.keys() != {DispatchKey.CompositeImplicitAutograd} and dispatch.keys() != {DispatchKey.CompositeImplicitAutogradNestedTensor} and (dispatch.keys() != {DispatchKey.CompositeImplicitAutograd, DispatchKey.CompositeImplicitAutogradNestedTensor})
        has_composite_implicit_autograd_kernel = DispatchKey.CompositeImplicitAutograd in dispatch.keys()
        has_composite_implicit_autograd_nested_tensor_kernel = DispatchKey.CompositeImplicitAutogradNestedTensor in dispatch.keys()
        has_composite_explicit_autograd_kernel = DispatchKey.CompositeExplicitAutograd in dispatch.keys()
        has_composite_explicit_autograd_non_functional_kernel = DispatchKey.CompositeExplicitAutogradNonFunctional in dispatch.keys()
        backend_metadata = {k: {func.name: v} for k, v in dispatch.items()}
        e.pop('__line__', None)
        assert not e, f'leftover entries: {e}'
        if structured_delegate is not None:
            for key in STRUCTURED_DISPATCH_KEYS:
                assert key not in dispatch, f'if structured_delegate, then must not have {key} in dispatch dictionary (it is delegated!)'
        return (NativeFunction(func=func, use_const_ref_for_mutable_tensors=use_const_ref_for_mutable_tensors, variants=variants, structured=structured, structured_delegate=structured_delegate, structured_inherits=structured_inherits, precomputed=precomputed, autogen=autogen, ufunc_inner_loop=ufunc_inner_loop, manual_kernel_registration=manual_kernel_registration, manual_cpp_binding=manual_cpp_binding, python_module=python_module, category_override=category_override, device_guard=device_guard, device_check=device_check, loc=loc, cpp_no_default_args=cpp_no_default_args, is_abstract=is_abstract, has_composite_implicit_autograd_kernel=has_composite_implicit_autograd_kernel, has_composite_implicit_autograd_nested_tensor_kernel=has_composite_implicit_autograd_nested_tensor_kernel, has_composite_explicit_autograd_kernel=has_composite_explicit_autograd_kernel, has_composite_explicit_autograd_non_functional_kernel=has_composite_explicit_autograd_non_functional_kernel, tags=tags, namespace=namespace), backend_metadata)

    def validate_unstructured(self) -> None:
        assert not self.structured, 'This function is structured, but there was no valid functional variant of it.'
        assert self.structured_delegate, 'This function delegates to another structured out function, but no valid function was found (the delegate may not exist, or it has the wrong type)'

    def __post_init__(self) -> None:
        if self.func.arguments.out:
            assert self.variants == {Variant.function}, 'Native functions with out arguments MUST be declared with only function variant; e.g., variants: function; otherwise you will tickle a Python argument binding bug (which usually manifests itself as the result variable being undefined.)'
        if self.structured:
            assert self.func.kind() == SchemaKind.out, 'Put structured field on the out= variant of a function; did you mean structured_delegate?'
            assert self.device_guard, 'device_guard: False is not respected by structured kernels'
        if self.structured_delegate:
            assert self.func.kind() != SchemaKind.out, 'structured_delegate field not allowed on out= functions; did you mean structured?'
            assert self.device_guard, 'device_guard: False is not respected by structured kernels'
        assert not (self.structured and self.structured_delegate), 'Cannot have both structured and structured_delegate on function'
        defaulted_arguments = {a.name for a in self.func.schema_order_arguments() if a.default is not None}
        invalid_args = set.difference(self.cpp_no_default_args, defaulted_arguments)
        assert len(invalid_args) == 0, f'Invalid cpp_no_default_args: {invalid_args}'
        if self.structured_inherits is not None:
            assert self.structured, 'structured_inherits must also imply structured: True'
        if str(self.func.name).startswith('_foreach'):
            assert self.device_check == DeviceCheckType.NoCheck, 'foreach kernels fall back to slow path when tensor are on different devices, device_check not allowed to be enabled'
        if 'rand' in str(self.func.name) or (('dropout' in str(self.func.name) or any(('dropout' in arg.name for arg in self.func.arguments.flat_all))) and 'backward' not in str(self.func.name) and (str(self.func.name.name) not in ['_cudnn_init_dropout_state'])) or self.func.arguments.has_generator_arg():
            assert 'nondeterministic_seeded' in self.tags, str(self.func.name)

    @property
    def has_composite_kernel(self) -> bool:
        return (self.has_composite_implicit_autograd_kernel or self.has_composite_explicit_autograd_kernel or self.has_composite_explicit_autograd_non_functional_kernel) or (self.has_composite_implicit_autograd_kernel and self.has_composite_implicit_autograd_nested_tensor_kernel)

    @property
    def is_view_op(self) -> bool:
        rets = self.func.returns
        is_non_mutating_view = len(rets) > 0 and any((r.annotation is not None and (not r.annotation.is_write) for r in rets))
        is_inplace_view = 'inplace_view' in self.tags and str(self.func.name) != 'resize_' and (str(self.func.name) != 'resize_as_')
        is_wildcard_view = any((inp.annotation is not None and '*' in inp.annotation.alias_set_after for inp in self.func.schema_order_arguments()))
        return is_non_mutating_view or is_inplace_view or is_wildcard_view

    @property
    def view_schema_kind(self) -> ViewSchemaKind:
        if self.is_view_op and self.func.name.name.inplace:
            assert 'inplace_view' in self.tags
            return ViewSchemaKind.aliasing_inplace
        if self.is_view_op:
            return ViewSchemaKind.aliasing
        else:
            return ViewSchemaKind.non_aliasing

    @property
    def root_name(self) -> str:
        return self.func.name.name.base

    @property
    def part_of_structured_group(self) -> bool:
        return self.structured or self.structured_delegate is not None