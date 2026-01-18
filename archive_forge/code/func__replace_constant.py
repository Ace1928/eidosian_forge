from typing import List, Optional, Union
import numpy as np
from onnx import (
from onnx.helper import (
from onnx.numpy_helper import from_array
def _replace_constant(node: NodeProto, threshold: int, value_constant_of_shape: float) -> List[NodeProto]:
    """Replaces a Constant node with a large tensor (with more than threshold elements) by a sequence of nodes that produces a dummy constant of same shape as original tensor."""
    if node.op_type != 'Constant':
        raise TypeError(f"Node type must be 'Constant' not {node.op_type!r}.")
    for att in node.attribute:
        if att.name == 'sparse_value':
            raise NotImplementedError(f'This feature is not yet implemented for a sparse constant (node name={node.name!r}).')
        if att.name == 'value':
            value = att.t
            new_name = f'{value.name}__SHAPE'
            dims = value.dims
            size = np.prod(dims)
            if size <= threshold:
                return [node]
            init = from_array(np.array(list(dims), dtype=np.int64), name=new_name)
            dtype = tensor_dtype_to_np_dtype(value.data_type)
            node_shape = make_node('Constant', [], [new_name], value=init)
            new_node = make_node('ConstantOfShape', [new_name], node.output, value=from_array(np.array([value_constant_of_shape], dtype=dtype)))
            return [node_shape, new_node]
        raise NotImplementedError(f'Replacement of constant with attribute {att.name!r}')
    return [node]