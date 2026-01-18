import itertools
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torchgen.api.dispatcher as dispatcher
from torchgen.api.lazy import (
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import method_with_native_function
from torchgen.dest.lazy_ts_lowering import ts_lowering_body
from torchgen.model import (
@dataclass(frozen=True)
class GenLazyNativeFuncDefinition:
    class_method_name: str
    backend_index: BackendIndex
    tensor_class: str
    gen_forced_fallback_code: bool
    backend_namespace: str
    get_tensorlist: str
    get_tensor_or_wrap_number: str
    try_get_tensor: str
    metrics_counter: str
    create_tensor: str
    create_from_first_tensor: bool
    create_aten_from_ltc_tensor: str
    tuple_aten_from_ltc_tensors: str
    lazy_tensor_ptr: str
    get_device_fn: str

    def lazy_tensor_decls(self, func: NativeFunction, schema: LazyIrSchema) -> str:
        value_args = schema.filtered_args(values=True, scalars=False)
        lazy_tensor_decls: List[str] = []
        for arg in value_args:
            if arg.is_wrapped_scalar:
                if isinstance(arg.lazy_type, OptionalCType):
                    lazy_tensor_decls.append(f'auto node_{arg.name} = {arg.name} ?\n                c10::make_optional(torch::lazy::LazyGraphExecutor::Get()->\n                    GetIrValueForScalarFromCodegen(*{arg.name}, *common_device)):\n                c10::nullopt;')
                else:
                    lazy_tensor_decls.append(f'auto node_{arg.name} = torch::lazy::LazyGraphExecutor::Get()->\n                            GetIrValueForScalarFromCodegen({arg.name}, *common_device);')
            elif arg.is_symint_or_list:
                continue
            elif isinstance(arg.lazy_type, BaseCType):
                if arg.lazy_type.type is tensorListValueT:
                    lazy_tensor_decls.append(f'auto lazy_{arg.name}_tensorlist = {self.backend_namespace}::{self.get_tensorlist}({arg.name});')
                else:
                    lazy_tensor_decls.append(f'{self.lazy_tensor_ptr} lazy_{arg.name} = {self.backend_namespace}::{self.get_tensor_or_wrap_number}({arg.name}, *common_device);')
            elif isinstance(arg.lazy_type, OptionalCType):
                assert arg.lazy_type.elem == BaseCType(getValueT()), arg.lazy_type.elem
                lazy_tensor_decls.append(f'{self.lazy_tensor_ptr} lazy_{arg.name} = {self.backend_namespace}::{self.try_get_tensor}({arg.name}.value_or(at::Tensor()));')
            else:
                raise AssertionError(f'TODO not sure if there are other valid types to handle here ({arg.lazy_type})')
        return '\n        '.join(lazy_tensor_decls)

    def force_eager_fallback(self, func: NativeFunction, schema: LazyIrSchema, metadata: BackendMetadata, sig: Union[DispatcherSignature, NativeSignature]) -> str:
        if self.gen_forced_fallback_code:
            return gen_fallback_code(schema, sig, overload_name=func.func.name.overload_name)
        return ''

    def metrics(self, func: NativeFunction, schema: LazyIrSchema) -> str:
        return f'{self.metrics_counter};'

    def get_device(self, func: NativeFunction, schema: LazyIrSchema) -> str:
        value_args = schema.filtered_args(values=True, scalars=False)
        scalar_args = schema.filtered_args(values=False, scalars=True)
        value_types_names = [f'{a.name}' for a in value_args if not a.is_wrapped_scalar]
        optional_device = OptionalCType(BaseCType(deviceT))
        optional_devices = [a.name for a in scalar_args if a.lazy_type == optional_device]
        assert len(value_types_names) > 0 or len(optional_devices) > 0, 'Expected at least one Value or Device type'
        get_device_str = f'{self.get_device_fn}({', '.join(value_types_names + optional_devices)})'
        return f'auto common_device = {get_device_str};\n        TORCH_INTERNAL_ASSERT(common_device);\n        '

    def shape_inference(self, func: NativeFunction, schema: LazyIrSchema) -> str:
        metadata = self.backend_index.get_kernel(func)
        assert metadata is not None
        all_args = schema.filtered_args()
        returns_length = len(schema.returns)
        is_view_copy_op = 'view_copy' in func.tags
        is_structured = func.structured or func.structured_delegate is not None
        if is_structured or is_view_copy_op:
            meta_out = '\nstd::vector<torch::lazy::Shape> shapes{torch::lazy::Shape(out_meta.scalar_type(), out_meta.sizes().vec())};'
            if returns_length > 1:

                def this_shape(i: int) -> str:
                    return f'torch::lazy::Shape(std::get<{i}>(out_meta).scalar_type(), std::get<{i}>(out_meta).sizes().vec())'
                shapes_str = ','.join([this_shape(i) for i in range(returns_length)])
                meta_out = 'std::vector<torch::lazy::Shape> shapes{' + shapes_str + '};'
            dispatcher_sig = DispatcherSignature.from_schema(func.func)
            meta_conversion_str, meta_call_ctx = convert_to_meta_tensors(dispatcher_sig)
            meta_call_args = [e.expr for e in translate(meta_call_ctx, dispatcher_sig.arguments(), method=False)]
            if is_view_copy_op:
                assert func.has_composite_explicit_autograd_non_functional_kernel
                dispatch_ns = 'compositeexplicitautogradnonfunctional'
            else:
                dispatch_ns = 'meta'
            aten_name = schema.aten_name
            if func.func.has_symint() and metadata.supports_symint():
                aten_name += '_symint'
            shape_str = f'        {meta_conversion_str}\n        auto out_meta = at::{dispatch_ns}::{aten_name}({', '.join(meta_call_args)});\n        {meta_out}'
        else:
            shape_sig = ComputeShapeSignature(metadata.kernel, func, symint=metadata.supports_symint())
            shape_str = f'\n            auto shapes = {shape_sig.shape_call};'
        shape_str += f'\n            TORCH_INTERNAL_ASSERT(shapes.size() == {returns_length});'
        func_schema_str = 'aten::' + str(func.func)
        shape_str += f'\n            if(torch::lazy::symbolicShapeEnabled()){{\n                std::vector<torch::jit::IValue> inputs = {{ {', '.join((str(a.name) for a in all_args))} }};\n                const char* schema_str = "{func_schema_str}";\n                applySymbolicShapesOnLT(schema_str, inputs, shapes);\n            }}\n        '
        return shape_str

    def build_ir_node(self, func: NativeFunction, schema: LazyIrSchema) -> str:
        node_ctor_input_str = node_ctor_inputs(schema)
        return f'torch::lazy::NodePtr node = torch::lazy::ReuseNode<{schema.node_name}>({node_ctor_input_str});\n        if (!node) {{\n            {self.shape_inference(func, schema)}\n            node = torch::lazy::MakeNode<{schema.node_name}>({node_ctor_input_str}, std::move(shapes));\n            CacheNode(node);\n        }}\n        '

    def create_lazy_tensor(self, first_tensor_name: Optional[str]=None) -> str:
        if self.create_from_first_tensor:
            assert first_tensor_name is not None, 'Requires first tensor to create lazy tensor'
            return f'{first_tensor_name}.{self.create_tensor}'
        return f'{self.backend_namespace}::{self.create_tensor}'

    def return_aten_tensor(self, func: NativeFunction, schema: LazyIrSchema) -> str:
        returns_length = len(schema.returns)
        value_args = schema.filtered_args(values=True, scalars=False)
        value_types_names = [f'{a.name}' for a in value_args if not a.is_wrapped_scalar]
        first_tensor_name = value_types_names[0] if len(value_types_names) > 0 else None
        bridge_str = f'auto result = {self.create_aten_from_ltc_tensor}(\n                {self.create_lazy_tensor(first_tensor_name)}(std::move(node), *common_device));'
        if returns_length > 1:
            assert len(value_types_names) > 0, 'Code below assumes there is at least one tensor arg'
            bridge_str = f'std::vector<{self.lazy_tensor_ptr}> lazy_tensors;\n        for (int i = 0; i < {returns_length}; i++) {{\n            lazy_tensors.push_back({self.create_lazy_tensor(first_tensor_name)}({getValueT()}(node, i), *common_device));\n        }}\n        auto result = {self.tuple_aten_from_ltc_tensors}<{returns_length}>(lazy_tensors);'
        if schema.name.name.inplace or func.func.is_out_fn():
            assert returns_length == 1, f'We assumed there was no such case where an op is an in-place variant and has tuple outputs, but got tuple of len {returns_length}.'
            bridge_str = f'lazy_{first_tensor_name}->SetInPlaceIrValue(node);\n        auto& result = {first_tensor_name};'
        bridge_str += '\n        return result;'
        return bridge_str

    @method_with_native_function
    def __call__(self, func: NativeFunction) -> List[str]:
        sig = kernel_signature(func, self.backend_index)
        metadata = self.backend_index.get_kernel(func)
        assert metadata is not None
        schema = LazyIrSchema(func.func, symint=metadata.supports_symint())
        return [f'    {sig.decl(name=f'{self.class_method_name}::{metadata.kernel}')} {{\n        {self.force_eager_fallback(func, schema, metadata, sig)}\n        {self.metrics(func, schema)}\n        {self.get_device(func, schema)}\n        {self.lazy_tensor_decls(func, schema)}\n        {self.build_ir_node(func, schema)}\n        {self.return_aten_tensor(func, schema)}\n    }}\n\n    ']