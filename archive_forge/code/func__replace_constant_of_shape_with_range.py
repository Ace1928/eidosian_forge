from typing import List, Optional, Union
import numpy as np
from onnx import (
from onnx.helper import (
from onnx.numpy_helper import from_array
def _replace_constant_of_shape_with_range(onx: Union[GraphProto, FunctionProto]) -> Union[GraphProto, FunctionProto]:
    """Replaces all *ConstantOfShape* by node *Range* to avoid constant tensors.

    The function is not recursive. The recursivity is done by
    *replace_initializer_by_constant_of_shape*.
    """
    if isinstance(onx, GraphProto):
        nodes = list(onx.node)
    elif isinstance(onx, FunctionProto):
        nodes = list(onx.node)
    else:
        raise TypeError(f'Not implemented for type {type(onx)}.')
    existing_names = set()
    for node in nodes:
        existing_names |= set(node.input)
        existing_names |= set(node.output)

    def _find_name(prefix):
        if prefix not in existing_names:
            existing_names.add(prefix)
            return prefix
        i = 2
        while True:
            name = f'{prefix}_{i}'
            if name not in existing_names:
                existing_names.add(name)
                return name
            i += 1
        raise RuntimeError('The function should never go through that line.')
    cst0 = make_node('Constant', [], [_find_name('zero')], value_int=0)
    cst1 = make_node('Constant', [], [_find_name('one')], value_int=1)
    update = {}
    for inode, node in enumerate(nodes):
        if node.op_type != 'ConstantOfShape':
            continue
        shape = node.input[0]
        n = make_node('ReduceProd', [shape], [_find_name(f'{shape}_N')])
        a = make_node('Range', [cst0.output[0], n.output[0], cst1.output[0]], [_find_name(f'{shape}_RANGE')])
        if len(node.attribute) == 1:
            to = node.attribute[0].t.data_type
        else:
            to = TensorProto.FLOAT
        ac = make_node('Cast', [a.output[0]], [_find_name(f'{shape}_RANGEf')], to=to)
        cl = make_node('Cast', [n.output[0]], [_find_name(f'{shape}_Nf')], to=to)
        d = make_node('Div', [ac.output[0], cl.output[0]], [_find_name(f'{shape}_FLAT')])
        resh = make_node('Reshape', [d.output[0], shape], node.output)
        update[inode] = [n, a, ac, cl, d, resh]
    for inode, up in sorted(update.items(), reverse=True):
        nodes[inode:inode + 1] = up
    nodes.insert(0, cst0)
    nodes.insert(1, cst1)
    if isinstance(onx, GraphProto):
        graph = make_graph(nodes, onx.name, onx.input, onx.output, initializer=onx.initializer, sparse_initializer=onx.sparse_initializer)
        return graph
    if isinstance(onx, FunctionProto):
        new_onx = make_function(onx.domain, onx.name, onx.input, onx.output, nodes, opset_imports=onx.opset_import)
        return new_onx
    raise TypeError(f'Not implemented for type {type(onx)}.')