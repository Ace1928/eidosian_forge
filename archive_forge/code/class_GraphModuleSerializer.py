import base64
import dataclasses
import io
import json
import logging
import math
import operator
import typing
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Union
import sympy
import torch
import torch.export.exported_program as ep
from torch._export.verifier import load_verifier
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental import symbolic_shapes
from torch.utils._pytree import treespec_dumps, treespec_loads
from torch.utils._sympy.value_ranges import ValueRanges
from .schema import (  # type: ignore[attr-defined]
from torch.export.exported_program import (
from .upgrade import GraphModuleOpUpgrader
class GraphModuleSerializer:

    def __init__(self, graph_signature: ep.ExportGraphSignature, module_call_graph: List[ep.ModuleCallEntry]):
        self.graph_state = GraphState()
        self.graph_signature = graph_signature
        self.module_call_graph = module_call_graph
        self.custom_objs: Dict[str, torch._C.ScriptObject] = {}

    @contextmanager
    def save_graph_state(self):
        saved = self.graph_state
        self.graph_state = GraphState()
        try:
            yield
        finally:
            self.graph_state = saved

    def handle_placeholder(self, node: torch.fx.Node):
        assert node.op == 'placeholder'
        if isinstance(node.meta['val'], torch.Tensor):
            graph_input = Argument.create(as_tensor=TensorArgument(name=node.name))
            self.graph_state.tensor_values[node.name] = serialize_tensor_meta(node.meta['val'])
        elif isinstance(node.meta['val'], torch.SymInt):
            raise AssertionError('SymInt graph input is not implemented yet.')
        elif isinstance(node.meta['val'], (int, bool, str, float, type(None))):
            graph_input = self.serialize_input(node.meta['val'])
        else:
            raise AssertionError(f'Unimplemented graph input type: {node.meta['val']}')
        self.graph_state.inputs.append(graph_input)

    def handle_output(self, node: torch.fx.Node):
        assert node.op == 'output'
        assert len(node.args) == 1, "FX.Node's args should have one arg"
        node_args = node.args[0]
        if isinstance(node_args, torch.fx.Node):
            self.graph_state.is_single_tensor_return = True
            self.graph_state.outputs = [self.serialize_input(node_args)]
        else:
            assert isinstance(node_args, (tuple, list))
            self.graph_state.outputs = [self.serialize_input(arg) for arg in node_args]

    def serialize_operator(self, target) -> str:
        if isinstance(target, str):
            return target
        elif target.__module__.startswith('torch._ops'):
            module = target.__module__.replace('torch._ops', 'torch.ops')
            return f'{module}.{target.__name__}'
        else:
            return f'{target.__module__}.{target.__name__}'

    def handle_call_function(self, node: torch.fx.Node):
        assert node.op == 'call_function'
        if node.target is operator.getitem:
            return
        if node.target in _SYM_INT_OPS:
            assert len(node.kwargs) == 0
            meta_val = node.meta['val']
            ex_node = Node(target=self.serialize_operator(node.target), inputs=self.serialize_sym_op_inputs(node.args), outputs=[Argument.create(as_sym_int=self.serialize_sym_int_output(node.name, meta_val))], metadata=self.serialize_metadata(node))
        elif node.target in _SYM_BOOL_OPS:
            assert len(node.kwargs) == 0
            meta_val = node.meta['val']
            ex_node = Node(target=self.serialize_operator(node.target), inputs=self.serialize_sym_op_inputs(node.args), outputs=[Argument.create(as_sym_bool=self.serialize_sym_bool_output(node.name, meta_val))], metadata=self.serialize_metadata(node))
        elif isinstance(node.target, torch._ops.OpOverload):
            ex_node = Node(target=self.serialize_operator(node.target), inputs=self.serialize_inputs(node.target, node.args, node.kwargs), outputs=self.serialize_outputs(node), metadata=self.serialize_metadata(node))
        elif isinstance(node.target, torch._ops.HigherOrderOperator):
            inputs = [NamedArgument(name='', arg=self.serialize_input(a)) for a in node.args]
            meta_val = node.meta['val']
            if isinstance(meta_val, torch.Tensor):
                outputs = [Argument.create(as_tensor=self.serialize_tensor_output(node.name, meta_val))]
            elif isinstance(meta_val, (list, tuple)) and all((isinstance(v, torch.Tensor) for v in meta_val)):
                arg_list = self._handle_getitem_users(node)
                outputs = [Argument.create(as_tensors=arg_list)]
            else:
                raise SerializeError('Only single tensor output or list of tensor output is supported for HigherOrderOperator serialization')
            ex_node = Node(target=self.serialize_operator(node.target), inputs=inputs, outputs=outputs, metadata=self.serialize_metadata(node))
        else:
            raise SerializeError(f'Serializing {node.target} is not supported')
        self.graph_state.nodes.append(ex_node)

    def handle_get_attr(self, node):
        pass

    def serialize_metadata(self, node: torch.fx.Node) -> Dict[str, str]:
        ret = {}
        if (stack_trace := node.meta.get('stack_trace')):
            ret['stack_trace'] = stack_trace
        if (nn_module_stack := node.meta.get('nn_module_stack')):

            def export_nn_module_stack(val):
                assert isinstance(val, tuple) and len(val) == 2
                path, ty = val
                assert isinstance(path, str)
                normalized_ty = ty.__module__ + '.' + ty.__qualname__
                return path + ',' + normalized_ty
            nn_module_list = [f'{k},{export_nn_module_stack(v)}' for k, v in nn_module_stack.items()]
            ret['nn_module_stack'] = ST_DELIMITER.join(nn_module_list)
        if (source_fn_st := node.meta.get('source_fn_stack')):
            source_fn_list = [f'{source_fn[0]},{self.serialize_operator(source_fn[1])}' for source_fn in source_fn_st]
            ret['source_fn_stack'] = ST_DELIMITER.join(source_fn_list)
        return ret

    def serialize_sym_op_inputs(self, args) -> List[NamedArgument]:
        serialized_args = []
        args_names = ['a', 'b']
        for args_name, arg in zip(args_names, args):
            serialized_args.append(NamedArgument(name=args_name, arg=self.serialize_input(arg)))
        return serialized_args

    def serialize_inputs(self, target: torch._ops.OpOverload, args, kwargs=None) -> List[NamedArgument]:
        assert isinstance(target, torch._ops.OpOverload)
        kwargs = kwargs or {}
        serialized_args = []
        for i, schema_arg in enumerate(target._schema.arguments):
            if schema_arg.name in kwargs:
                serialized_args.append(NamedArgument(name=schema_arg.name, arg=self.serialize_input(kwargs[schema_arg.name])))
            elif not schema_arg.kwarg_only and i < len(args):
                serialized_args.append(NamedArgument(name=schema_arg.name, arg=self.serialize_input(args[i])))
            else:
                pass
        return serialized_args

    def is_sym_int_arg(self, arg) -> bool:
        return isinstance(arg, int) or (isinstance(arg, torch.fx.Node) and arg.name in self.graph_state.sym_int_values)

    def is_sym_bool_arg(self, arg) -> bool:
        return isinstance(arg, bool) or (isinstance(arg, torch.fx.Node) and arg.name in self.graph_state.sym_bool_values)

    def serialize_input(self, arg) -> Argument:
        import torch._inductor.ir as inductor_ir
        inductor_tensor_buffers = (inductor_ir.Buffer, inductor_ir.ReinterpretView)
        if isinstance(arg, torch.fx.Node):
            if arg.op == 'get_attr':
                assert isinstance(arg.target, str)
                attr = getattr(arg.graph.owning_module, arg.target)
                if isinstance(attr, torch.Tensor):
                    raise SerializeError('getattr nodes containing tensors should not appear in the graph')
                elif isinstance(attr, torch.fx.GraphModule):
                    with self.save_graph_state():
                        graph = self.serialize_graph(attr)
                    return Argument.create(as_graph=GraphArgument(name=arg.target, graph=graph))
                else:
                    raise SerializeError(f'Unsupported getattr attribute {arg.target} with type: {type(attr)}')
            elif self.is_sym_int_arg(arg):
                return Argument.create(as_sym_int=SymIntArgument.create(as_name=arg.name))
            elif self.is_sym_bool_arg(arg):
                return Argument.create(as_sym_bool=SymBoolArgument.create(as_name=arg.name))
            else:
                return Argument.create(as_tensor=TensorArgument(name=arg.name))
        elif isinstance(arg, inductor_tensor_buffers):
            arg_name = arg.get_name()
            assert arg_name is not None, 'Buffer must have valid name'
            return Argument.create(as_tensor=TensorArgument(name=arg_name))
        elif isinstance(arg, torch.SymInt):
            return Argument.create(as_sym_int=SymIntArgument.create(as_name=str(arg)))
        elif isinstance(arg, bool):
            return Argument.create(as_bool=arg)
        elif isinstance(arg, str):
            return Argument.create(as_string=arg)
        elif isinstance(arg, int):
            return Argument.create(as_int=arg)
        elif isinstance(arg, float):
            return Argument.create(as_float=arg)
        elif arg is None:
            return Argument.create(as_none=())
        elif isinstance(arg, (list, tuple)):
            if all((isinstance(a, bool) for a in arg)):
                return Argument.create(as_bools=list(arg))
            elif all((isinstance(a, int) for a in arg)):
                return Argument.create(as_ints=list(arg))
            elif all((isinstance(a, float) for a in arg)):
                return Argument.create(as_floats=list(arg))
            elif all((isinstance(a, str) for a in arg)):
                return Argument.create(as_strings=list(arg))
            elif all((isinstance(a, torch.SymInt) for a in arg)):
                return Argument.create(as_sym_ints=[SymIntArgument.create(as_name=str(a)) for a in arg])
            elif all((self.is_sym_int_arg(a) for a in arg)):
                values = []
                for a in arg:
                    if isinstance(a, torch.fx.Node):
                        values.append(SymIntArgument.create(as_name=a.name))
                    elif isinstance(a, int):
                        values.append(SymIntArgument.create(as_int=a))
                return Argument.create(as_sym_ints=values)
            elif all((self.is_sym_bool_arg(a) for a in arg)):
                values = []
                for a in arg:
                    if isinstance(a, torch.fx.Node):
                        values.append(SymBoolArgument.create(as_name=a.name))
                    elif isinstance(a, bool):
                        values.append(SymBoolArgument.create(as_bool=a))
                return Argument.create(as_sym_bools=values)
            elif all((isinstance(a, torch.fx.Node) for a in arg)):
                arguments = []
                for a in arg:
                    if a.op == 'get_attr':
                        raise SerializeError('getattr nodes containing tensors should not appear in the graph')
                    arguments.append(TensorArgument(name=a.name))
                return Argument.create(as_tensors=arguments)
            elif all((isinstance(a, (torch.fx.Node, type(None))) for a in arg)):

                def serialize_optional_tensor_args(a):
                    if a is None:
                        return OptionalTensorArgument.create(as_none=())
                    elif isinstance(a, torch.fx.Node):
                        return OptionalTensorArgument.create(as_tensor=a.name)
                    else:
                        raise SerializeError(f'Unsupported list/tuple argument: {a}')
                return Argument.create(as_optional_tensors=list(map(serialize_optional_tensor_args, arg)))
            elif all((isinstance(a, inductor_tensor_buffers) for a in arg)):
                return Argument.create(as_tensors=[TensorArgument(name=a.get_name()) for a in arg])
            elif all((isinstance(a, (*inductor_tensor_buffers, type(None))) for a in arg)):

                def serialize_optional_tensor_args(a):
                    if a is None:
                        return OptionalTensorArgument.create(as_none=())
                    elif isinstance(a, inductor_tensor_buffers):
                        return OptionalTensorArgument.create(as_tensor=a.get_name())
                    else:
                        raise SerializeError(f'Unsupported list/tuple argument: {a}')
                return Argument.create(as_optional_tensors=list(map(serialize_optional_tensor_args, arg)))
            else:
                raise SerializeError(f'Unsupported list/tuple argument type: {[type(a) for a in arg]}')
        elif isinstance(arg, torch.dtype):
            return Argument.create(as_scalar_type=_TORCH_TO_SERIALIZE_DTYPE[arg])
        elif isinstance(arg, torch.device):
            return Argument.create(as_device=Device(type=arg.type, index=arg.index))
        elif isinstance(arg, torch.memory_format):
            return Argument.create(as_memory_format=_TORCH_TO_SERIALIZE_MEMORY_FORMAT[arg])
        elif isinstance(arg, torch.layout):
            return Argument.create(as_layout=_TORCH_TO_SERIALIZE_LAYOUT[arg])
        elif isinstance(arg, torch._C.ScriptObject):
            if not (arg._has_method('__getstate__') and arg._has_method('__setstate__')):
                raise SerializeError(f'Unable to serialize custom class {arg}. Please define serialization methods via def_pickle().')
            custom_obj_name = f'_custom_obj_{len(self.custom_objs)}'
            self.custom_objs[custom_obj_name] = arg
            return Argument.create(as_custom_obj=CustomObjArgument(custom_obj_name))
        else:
            raise SerializeError(f'Unsupported argument type: {type(arg)}')

    def serialize_tensor_output(self, name, meta_val) -> TensorArgument:
        assert name not in self.graph_state.tensor_values
        self.graph_state.tensor_values[name] = serialize_tensor_meta(meta_val)
        return TensorArgument(name=name)

    def serialize_sym_int_output(self, name, meta_val) -> SymIntArgument:
        assert name not in self.graph_state.sym_int_values
        self.graph_state.sym_int_values[name] = serialize_sym_int(meta_val)
        return SymIntArgument.create(as_name=name)

    def serialize_sym_bool_output(self, name, meta_val) -> SymIntArgument:
        assert name not in self.graph_state.sym_bool_values
        self.graph_state.sym_bool_values[name] = serialize_sym_bool(meta_val)
        return SymBoolArgument.create(as_name=name)

    def serialize_input_spec(self, spec: ep.InputSpec) -> InputSpec:
        if spec.kind == ep.InputKind.USER_INPUT:
            return InputSpec.create(user_input=UserInputSpec(arg=self.serialize_argument_spec(spec.arg)))
        elif spec.kind == ep.InputKind.PARAMETER:
            assert spec.target is not None
            assert isinstance(spec.arg, ep.TensorArgument)
            return InputSpec.create(parameter=InputToParameterSpec(arg=TensorArgument(name=spec.arg.name), parameter_name=spec.target))
        elif spec.kind == ep.InputKind.BUFFER:
            assert spec.target is not None
            assert isinstance(spec.arg, ep.TensorArgument)
            return InputSpec.create(buffer=InputToBufferSpec(arg=TensorArgument(name=spec.arg.name), buffer_name=spec.target))
        elif spec.kind == ep.InputKind.CONSTANT_TENSOR:
            assert spec.target is not None
            assert isinstance(spec.arg, ep.TensorArgument)
            return InputSpec.create(tensor_constant=InputToTensorConstantSpec(arg=TensorArgument(name=spec.arg.name), tensor_constant_name=spec.target))
        else:
            raise AssertionError(f'Unknown argument kind: {spec}')

    def serialize_output_spec(self, spec: ep.OutputSpec) -> OutputSpec:
        if spec.kind == ep.OutputKind.USER_OUTPUT:
            return OutputSpec.create(user_output=UserOutputSpec(arg=self.serialize_argument_spec(spec.arg)))
        elif spec.kind == ep.OutputKind.LOSS_OUTPUT:
            assert isinstance(spec.arg, ep.TensorArgument)
            return OutputSpec.create(loss_output=LossOutputSpec(arg=TensorArgument(name=spec.arg.name)))
        elif spec.kind == ep.OutputKind.BUFFER_MUTATION:
            assert spec.target is not None
            assert isinstance(spec.arg, PyTensorArgument)
            return OutputSpec.create(buffer_mutation=BufferMutationSpec(arg=TensorArgument(name=spec.arg.name), buffer_name=spec.target))
        elif spec.kind == ep.OutputKind.GRADIENT_TO_PARAMETER:
            assert spec.target is not None
            assert isinstance(spec.arg, PyTensorArgument)
            return OutputSpec.create(gradient_to_parameter=GradientToParameterSpec(arg=TensorArgument(name=spec.arg.name), parameter_name=spec.target))
        elif spec.kind == ep.OutputKind.GRADIENT_TO_USER_INPUT:
            assert spec.target is not None
            assert isinstance(spec.arg, PyTensorArgument)
            return OutputSpec.create(gradient_to_user_input=GradientToUserInputSpec(arg=TensorArgument(name=spec.arg.name), user_input_name=spec.target))
        else:
            raise AssertionError(f'Unknown argument kind: {spec}')

    def serialize_signature(self, sig: ep.ExportGraphSignature) -> GraphSignature:
        return GraphSignature(input_specs=[self.serialize_input_spec(s) for s in sig.input_specs], output_specs=[self.serialize_output_spec(s) for s in sig.output_specs])

    def serialize_argument_spec(self, x: ep.ArgumentSpec) -> Argument:
        if isinstance(x, PyTensorArgument):
            return Argument.create(as_tensor=TensorArgument(name=x.name))
        elif isinstance(x, PySymIntArgument):
            return Argument.create(as_sym_int=SymIntArgument.create(as_name=x.name))
        elif isinstance(x, PyConstantArgument):
            return self.serialize_input(x.value)
        else:
            raise AssertionError('TODO')

    def serialize_module_call_signature(self, module_call_signature: ep.ModuleCallSignature) -> ModuleCallSignature:
        return ModuleCallSignature(inputs=[self.serialize_argument_spec(x) for x in module_call_signature.inputs], outputs=[self.serialize_argument_spec(x) for x in module_call_signature.outputs], in_spec=treespec_dumps(module_call_signature.in_spec, TREESPEC_VERSION), out_spec=treespec_dumps(module_call_signature.out_spec, TREESPEC_VERSION))

    def serialize_module_call_graph(self, module_call_graph: List[ep.ModuleCallEntry]) -> List[ModuleCallEntry]:
        return [ModuleCallEntry(fqn=entry.fqn, signature=self.serialize_module_call_signature(entry.signature) if entry.signature else None) for entry in module_call_graph]

    def serialize_outputs(self, node: torch.fx.Node) -> List[Argument]:
        """For a given node, return the dataclass representing its output values.

        [NOTE: Multiple outputs] We handle aggregates differently than FX. For
        FX, it looks like:

            x = call_function("multiple_return", ...)
            element0 = call_function(getitem, x, 0)
            foo = call_function("use_output", element0)

        We do not want the intermediate `getitem` call, so our serialized thing looks like:

            element0, element1, element2 = call_function("multiple_return", ...)
            foo = call_function("use_output", element0)

        We want names to be consistent across these two schemes, so that we can
        mostly reuse the names coming from FX. This function computes a mapping from
        the FX representation to our representation, preserving the names.
        """
        assert node.op == 'call_function' and isinstance(node.target, torch._ops.OpOverload)
        assert isinstance(node.target, torch._ops.OpOverload)
        returns = node.target._schema.returns
        if len(returns) == 0:
            return []
        meta_val = node.meta['val']

        def output_node_at_index(node, index):
            for user in node.users:
                assert user.target is operator.getitem, f'{user} is not a getitem node'
                if index == user.args[1]:
                    return user
            return None
        if _is_single_tensor_return(node.target):
            return [Argument.create(as_tensor=self.serialize_tensor_output(node.name, meta_val))]
        elif len(returns) == 1 and isinstance(meta_val, torch.SymInt):
            return [Argument.create(as_sym_int=self.serialize_sym_int_output(node.name, meta_val))]
        elif len(returns) == 1 and isinstance(meta_val, torch.SymBool):
            return [Argument.create(as_sym_bool=self.serialize_sym_bool_output(node.name, meta_val))]
        elif _is_single_tensor_list_return(node.target):
            tensor_args = []
            for idx, meta in enumerate(meta_val):
                user_node = output_node_at_index(node, idx)
                name = user_node.name if user_node is not None else f'{node.name}_unused_{idx}'
                tensor_args.append(self.serialize_tensor_output(name, meta))
            return [Argument.create(as_tensors=tensor_args)]
        output_arguments = []
        for idx, (meta, return_schema) in enumerate(zip(meta_val, returns)):
            if meta is None:
                assert isinstance(return_schema.real_type, torch.OptionalType)
                output_arguments.append(Argument.create(as_none=()))
            elif isinstance(meta, torch._subclasses.fake_tensor.FakeTensor):
                assert isinstance(return_schema.real_type, torch.TensorType)
                user_node = output_node_at_index(node, idx)
                name = user_node.name if user_node is not None else f'{node.name}_unused_{idx}'
                output_arguments.append(Argument.create(as_tensor=self.serialize_tensor_output(name, meta)))
            elif isinstance(meta, list):
                assert isinstance(return_schema.real_type, torch.ListType) and isinstance(return_schema.real_type.getElementType(), torch.TensorType)
                user_node = output_node_at_index(node, idx)
                assert user_node is not None
                args = []
                for i, m in enumerate(meta):
                    if m is None:
                        continue
                    sub_user_node = output_node_at_index(user_node, i)
                    assert sub_user_node is not None, f'No user found at index {i}'
                    args.append(self.serialize_tensor_output(sub_user_node.name, m))
                output_arguments.append(Argument.create(as_tensors=args))
        return output_arguments

    def _handle_getitem_users(self, node: torch.fx.Node) -> List[TensorArgument]:
        meta_val = node.meta['val']
        idx_to_name = {}
        for user in node.users:
            assert user.target is operator.getitem, f'User node {user} of {node} is incorrect'
            idx_to_name[user.args[1]] = user.name
        for idx, _ in enumerate(meta_val):
            if idx not in idx_to_name:
                idx_to_name[idx] = f'{node.name}_unused_{idx}'
        arg_list = []
        for i, element_meta_val in enumerate(meta_val):
            arg_list.append(self.serialize_tensor_output(idx_to_name[i], element_meta_val))
        return arg_list

    def serialize_graph(self, graph_module: torch.fx.GraphModule) -> Graph:
        assert isinstance(graph_module, torch.fx.GraphModule)
        for node in graph_module.graph.nodes:
            try:
                getattr(self, f'handle_{node.op}')(node)
            except Exception as e:
                raise SerializeError(f'Failed serializing node {node} in graph: {node.format_node()}') from e
        return Graph(inputs=self.graph_state.inputs, nodes=self.graph_state.nodes, tensor_values=self.graph_state.tensor_values, sym_int_values=self.graph_state.sym_int_values, sym_bool_values=self.graph_state.sym_bool_values, outputs=self.graph_state.outputs, is_single_tensor_return=self.graph_state.is_single_tensor_return)

    def serialize(self, graph_module: torch.fx.GraphModule) -> GraphModule:
        graph = self.serialize_graph(graph_module)
        return GraphModule(graph=graph, signature=self.serialize_signature(self.graph_signature), module_call_graph=self.serialize_module_call_graph(self.module_call_graph))