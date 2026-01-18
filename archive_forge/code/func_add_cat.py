import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def add_cat(self, node):
    assert node.inputsSize() == 2
    assert node.outputsSize() == 1
    tensors = self.tensor_sequences[node.inputsAt(0)]
    _, dim = self.get_constant_value(node.inputsAt(1), 'IntType')
    assert len(tensors) > 0
    in_ids = []
    out_oper = None
    out_dim_size = 0
    for inp in tensors:
        in_id, in_oper = self.get_tensor_operand_by_jitval(inp)
        if out_oper is None:
            out_shape = change_element(in_oper.shape, dim, -1)
            out_oper = in_oper._replace(shape=out_shape)
        assert in_oper.op_type == out_oper.op_type
        assert in_oper.dim_order == out_oper.dim_order
        assert change_element(in_oper.shape, dim, -1) == change_element(out_oper.shape, dim, -1)
        in_ids.append(in_id)
        out_dim_size += in_oper.shape[dim]
    assert out_oper is not None
    out_oper = out_oper._replace(shape=change_element(out_oper.shape, dim, out_dim_size))
    if in_oper.dim_order == DimOrder.CHANNELS_LAST:
        assert len(out_oper.shape) == 4
        nnapi_dim = [0, 3, 1, 2][dim]
    else:
        nnapi_dim = dim
    out_id = self.add_tensor_operand(node.outputsAt(0), out_oper)
    for idx, d in enumerate(out_oper.shape):
        if d == 0:
            if idx == dim:
                shape = ' + '.join((flex_name(ip_id, dim) for ip_id in in_ids))
                self.compute_operand_shape(out_id, idx, shape)
            else:
                self.forward_operand_shape(out_id, idx, in_ids[0], idx)
    inputs = in_ids + [self.add_immediate_int_scalar(nnapi_dim)]
    outputs = [None] * 1
    outputs[0] = out_id
    self.add_operation(NNAPI_OperationCode.CONCATENATION, inputs, outputs)