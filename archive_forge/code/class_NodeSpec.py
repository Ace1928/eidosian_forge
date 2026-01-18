import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.intrinsic.quantized.dynamic as nniqd
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
from torch.ao.nn.intrinsic import _FusedModule
import torch.distributed as dist
from torch.testing._internal.common_utils import TestCase, TEST_WITH_ROCM
from torch.ao.quantization import (
from torch.ao.quantization import QuantWrapper, QuantStub, DeQuantStub, \
from torch.ao.quantization.quantization_mappings import (
from torch.testing._internal.common_quantized import (
from torch.jit.mobile import _load_for_lite_interpreter
import copy
import io
import functools
import time
import os
import unittest
import numpy as np
from torch.testing import FileCheck
from typing import Callable, Tuple, Dict, Any, Union, Type, Optional
import torch._dynamo as torchdynamo
class NodeSpec:
    """ Used for checking GraphModule Node
    """

    def __init__(self, op, target):
        """
        op: call_function | call_module
        target:
          for call_function, target would be a function
          for call_module, target would be the type of PyTorch module
        """
        self.op = op
        self.target = target

    @classmethod
    def call_function(cls, target):
        return NodeSpec('call_function', target)

    @classmethod
    def call_method(cls, target):
        return NodeSpec('call_method', target)

    @classmethod
    def call_module(cls, target):
        return NodeSpec('call_module', target)

    def __hash__(self):
        return hash((self.op, self.target))

    def __eq__(self, other):
        if not isinstance(other, NodeSpec):
            return NotImplemented
        return self.op == other.op and self.target == other.target

    def __repr__(self):
        return repr(self.op) + ' ' + repr(self.target)