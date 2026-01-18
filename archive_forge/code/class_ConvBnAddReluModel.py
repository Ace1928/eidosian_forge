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
class ConvBnAddReluModel(torch.nn.Module):

    def __init__(self, with_bn=True, with_relu=True, left_conv=True, two_conv=True, use_torch_add=True):
        super().__init__()
        self.conv = nn.Conv2d(5, 5, (2, 2))
        self.conv2 = nn.Conv2d(5, 5, (2, 2))
        self.bn = nn.BatchNorm2d(5)
        self.relu = nn.ReLU()
        self.with_bn = with_bn
        self.with_relu = with_relu
        self.two_conv = two_conv
        self.left_conv = left_conv
        self.use_torch_add = use_torch_add

    def forward(self, x1, x2):
        if self.two_conv:
            if self.use_torch_add:
                if self.with_bn:
                    x = torch.add(self.bn(self.conv(x1)), self.conv2(x1))
                else:
                    x = torch.add(self.conv(x1), self.conv2(x1))
            elif self.with_bn:
                x = self.bn(self.conv(x1)) + self.conv2(x1)
            else:
                x = self.conv(x1) + self.conv2(x1)
        elif self.use_torch_add:
            if self.left_conv:
                if self.with_bn:
                    x = torch.add(self.bn(self.conv(x1)), x2)
                else:
                    x = torch.add(self.conv(x1), x2)
            elif self.with_bn:
                x = torch.add(x2, self.bn(self.conv(x1)))
            else:
                x = torch.add(x2, self.conv(x1))
        elif self.left_conv:
            if self.with_bn:
                x = self.bn(self.conv(x1)) + x2
            else:
                x = self.conv(x1) + x2
        elif self.with_bn:
            x = x2 + self.bn(self.conv(x1))
        else:
            x = x2 + self.conv(x1)
        if self.with_relu:
            x = self.relu(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return (torch.rand(1, 5, 3, 3), torch.rand(1, 5, 2, 2))