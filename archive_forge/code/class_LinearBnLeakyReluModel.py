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
class LinearBnLeakyReluModel(torch.nn.Module):

    def __init__(self, with_bn=True):
        super().__init__()
        self.linear = nn.Linear(5, 5)
        self.bn1d = nn.BatchNorm1d(5)
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.with_bn = with_bn

    def forward(self, x):
        x = self.linear(x)
        if self.with_bn:
            x = self.bn1d(x)
        x = self.leaky_relu(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return (torch.rand(1, 5),)