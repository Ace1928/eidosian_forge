import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def add_dequantize(self, node):
    assert node.inputsSize() == 1
    assert node.outputsSize() == 1
    in_id, in_oper = self.get_tensor_operand_by_jitval_fixed_size(node.inputsAt(0))
    out_oper = in_oper._replace(op_type=NNAPI_OperandCode.TENSOR_FLOAT32, scale=0.0, zero_point=0)
    inputs = [None] * 1
    inputs[0] = in_id
    outputs = [None] * 1
    outputs[0] = self.add_tensor_operand(node.outputsAt(0), out_oper)
    self.add_operation(NNAPI_OperationCode.DEQUANTIZE, inputs, outputs)