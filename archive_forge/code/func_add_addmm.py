import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def add_addmm(self, node):
    assert node.inputsSize() == 5
    assert node.outputsSize() == 1
    jit_bias, jit_input, jit_weight, jit_beta, jit_alpha = node.inputs()
    for jitval in (jit_beta, jit_alpha):
        scale_ctype, scale_value = self.get_constant_value(jitval)
        assert scale_ctype.kind() in ('IntType', 'FloatType')
        if scale_value != 1:
            raise Exception('NNAPI Fully-Connected does not support alpha and beta.')
    self.add_addmm_or_linear(node, True, jit_input, jit_weight, jit_bias)