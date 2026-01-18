import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def _do_add_binary(self, node, opcode, fuse_code, *, qparams=None):
    """Helper for pointwise binary broadcast ops with superfluous extra args."""
    assert node.outputsSize() == 1
    assert node.inputsAt(0).type().kind() == 'TensorType'
    assert node.inputsAt(1).type().kind() == 'TensorType'
    if self.has_operand_for_jitval(node.inputsAt(0)):
        in0_id, in0_oper = self.get_tensor_operand_by_jitval(node.inputsAt(0))
        in1_id, in1_oper = self.get_tensor_operand_or_constant(node.inputsAt(1), in0_oper.dim_order)
    elif self.has_operand_for_jitval(node.inputsAt(1)):
        in1_id, in1_oper = self.get_tensor_operand_by_jitval(node.inputsAt(1))
        in0_id, in0_oper = self.get_tensor_operand_or_constant(node.inputsAt(0), in1_oper.dim_order)
    else:
        raise Exception(f"Can't do a NNAPI binary op: {opcode} on two constants")
    assert in0_oper.op_type == in1_oper.op_type
    in0_id, in0_oper, in1_id, in1_oper = self.transpose_for_broadcast(in0_id, in0_oper, in1_id, in1_oper)
    out_shape = broadcast_shapes(in0_oper.shape, in1_oper.shape)
    out_oper = in0_oper._replace(shape=out_shape)
    if qparams is not None:
        scale, zp = qparams
        out_oper = out_oper._replace(scale=scale, zero_point=zp)
    out_id = self.add_tensor_operand(node.outputsAt(0), out_oper)
    for idx, (d0, d1) in enumerate(zip(in0_oper.shape, in1_oper.shape)):
        if d0 == 1 and d1 == 0:
            self.forward_operand_shape(out_id, idx, in1_id, idx)
        elif d0 == 0 and d1 == 1:
            self.forward_operand_shape(out_id, idx, in0_id, idx)
        elif d0 == 0 and d1 == 0:
            self.flexible_shape_computation_lines.append(f'assert {flex_name(in0_id, idx)} == {flex_name(in1_id, idx)}')
            self.forward_operand_shape(out_id, idx, in0_id, idx)
    inputs = [None] * 3
    inputs[0] = in0_id
    inputs[1] = in1_id
    inputs[2] = self.add_immediate_int_scalar(fuse_code)
    outputs = [None] * 1
    outputs[0] = out_id
    self.add_operation(opcode, inputs, outputs)