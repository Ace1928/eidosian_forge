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
class ConvBnReLU2dAndLinearReLU(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_bn_relu = TestHelperModules.ConvWithBNRelu(relu=True)
        self.linear = torch.nn.Linear(3, 8, bias=False)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv_bn_relu(x)
        permute_out = torch.permute(x, (0, 2, 3, 1))
        linear_out = self.linear(permute_out)
        return linear_out