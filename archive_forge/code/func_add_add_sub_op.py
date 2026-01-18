import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def add_add_sub_op(self, node, opcode, fuse_code):
    assert node.inputsSize() == 3
    _, alpha = self.get_constant_value(node.inputsAt(2), 'IntType')
    if alpha != 1:
        raise Exception('NNAPI does not support add/sub with alpha.')
    self._do_add_binary(node, opcode, fuse_code)