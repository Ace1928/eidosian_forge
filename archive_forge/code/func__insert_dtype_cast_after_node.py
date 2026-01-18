import torch
from torch.fx import GraphModule, map_arg
from torch.fx.graph import Graph, Node
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from .utils import (
from .ns_types import (
from torch.ao.ns.fx.mappings import (
from torch.ao.quantization.observer import _is_activation_post_process
from typing import Dict, Tuple, Callable, List, Any, Union, Optional, Set
def _insert_dtype_cast_after_node(node_a: Node, node_c: Node, prev_node_c: Union[Node, List[Node]], gm_a: GraphModule, gm_b: GraphModule, graph_c: Graph, node_name_prefix: str, logger_cls: Callable, node_type_to_io_type_map: Dict[str, Set[NSNodeTargetType]]) -> Union[Node, List[Node]]:
    """
    Given a starting graph C (derived from graph B) of

    ... -> prev_node_c -> node_c -> ...

    And a corresponding related node_a, inserts the correct dtype
    cast node after prev_node_c to cast into the dtype expected
    by node_a, resulting in:

                          dtype_cast
                        /
    ... -> prev_node_c -> node_c -> ...

    For example, if node_c is an int8 op and node_a is an fp32 op, this function
    will insert a dequant.
    """
    dtype_cast_op = None
    dtype_cast_mod_cls = None
    dtype_cast_method = None
    dtype_cast_method_dtype = None
    dtype_cast_scale = None
    dtype_cast_zero_point = None
    node_input_type_a, _node_output_type_a = get_node_first_input_and_output_type(node_a, gm_a, logger_cls, node_type_to_io_type_map)
    node_input_type_c, _node_output_type_c = get_node_first_input_and_output_type(node_c, gm_b, logger_cls, node_type_to_io_type_map)
    if node_input_type_a == NodeInputOrOutputType.FP32 and node_input_type_c == NodeInputOrOutputType.INT8 or (node_input_type_a == NodeInputOrOutputType.FP32 and node_input_type_c == NodeInputOrOutputType.FP16) or (node_input_type_a == NodeInputOrOutputType.FP32 and node_input_type_c == NodeInputOrOutputType.FP32_OR_INT8):
        dtype_cast_op = torch.dequantize
    elif node_input_type_a == node_input_type_c and node_input_type_a != NodeInputOrOutputType.UNKNOWN:
        dtype_cast_mod_cls = torch.nn.Identity
    elif node_input_type_a == NodeInputOrOutputType.INT8 and node_input_type_c == NodeInputOrOutputType.FP32:
        node_a_input_qparams = get_node_input_qparams(node_a, gm_a, node_type_to_io_type_map)
        if node_a_input_qparams is not None:
            dtype_cast_op = torch.quantize_per_tensor
            dtype_cast_scale, dtype_cast_zero_point = node_a_input_qparams
    elif node_input_type_a == NodeInputOrOutputType.FP16 and node_input_type_c == NodeInputOrOutputType.FP32:
        dtype_cast_method = 'to'
        dtype_cast_method_dtype = torch.float16
    else:
        raise AssertionError(f'dtype cast from {node_input_type_c} {node_c.format_node()} to ' + f'{node_input_type_a} {node_a.format_node()} needs to be implemented')
    if isinstance(prev_node_c, Node):
        new_dtype_cast_name = get_new_attr_name_with_prefix(node_name_prefix)(gm_b)
        if dtype_cast_op:
            if dtype_cast_scale is not None and dtype_cast_zero_point is not None:
                return _insert_quantize_per_tensor_node(prev_node_c, node_a, gm_b, graph_c, dtype_cast_scale, dtype_cast_zero_point, new_dtype_cast_name)
            else:
                return graph_c.create_node('call_function', dtype_cast_op, (prev_node_c,), {}, new_dtype_cast_name)
        elif dtype_cast_method:
            return graph_c.create_node('call_method', dtype_cast_method, (prev_node_c, dtype_cast_method_dtype), {}, new_dtype_cast_name)
        else:
            assert dtype_cast_mod_cls
            dtype_cast_mod = dtype_cast_mod_cls()
            setattr(gm_b, new_dtype_cast_name, dtype_cast_mod)
            return graph_c.create_node('call_module', new_dtype_cast_name, (prev_node_c,), {}, new_dtype_cast_name)
    elif isinstance(prev_node_c, list):
        results = []
        for prev_node_c_inner in prev_node_c:
            new_dtype_cast_name = get_new_attr_name_with_prefix(node_name_prefix)(gm_b)
            if dtype_cast_op:
                new_dtype_cast_node = graph_c.create_node('call_function', dtype_cast_op, (prev_node_c_inner,), {}, new_dtype_cast_name)
                results.append(new_dtype_cast_node)
            else:
                assert dtype_cast_mod_cls
                dtype_cast_mod = dtype_cast_mod_cls()
                setattr(gm_b, new_dtype_cast_name, dtype_cast_mod)
                new_dtype_cast_node = graph_c.create_node('call_module', new_dtype_cast_name, (prev_node_c_inner,), {}, new_dtype_cast_name)
                results.append(new_dtype_cast_node)
        return results
    else:
        raise AssertionError(f'type f{type(prev_node_c)} is not handled')