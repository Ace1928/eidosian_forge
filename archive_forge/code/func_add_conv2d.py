import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def add_conv2d(self, node):
    assert node.inputsSize() == 7
    assert node.outputsSize() == 1
    jit_image, jit_weight, jit_bias, jit_stride, jit_pad, jit_dilation, jit_groups = node.inputs()
    _, weight_tensor = self.get_constant_value(jit_weight, 'TensorType')
    bias_id, bias_oper = self.get_optional_bias(jit_bias, weight_tensor)
    args = self.get_conv_pool_args_2d_from_jit(weight_tensor.shape[2:4], jit_stride, jit_pad, jit_dilation, jit_groups)
    return self.add_conv2d_common(node.outputsAt(0), 0.0, 0, jit_image, weight_tensor, bias_id, args, False, NNAPI_FuseCode.FUSED_NONE)