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
class NormalizationTestModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.fc1 = torch.nn.Linear(5, 8).to(dtype=torch.float)
        self.layer_norm = torch.nn.LayerNorm(8)
        self.group_norm = torch.nn.GroupNorm(2, 8)
        self.instance_norm1d = torch.nn.InstanceNorm1d(8)
        self.instance_norm2d = torch.nn.InstanceNorm2d(8)
        self.instance_norm3d = torch.nn.InstanceNorm3d(8)

    def forward(self, x):
        x = self.quant(x)
        x = self.fc1(x)
        x = self.layer_norm(x)
        x = self.group_norm(x.unsqueeze(-1).repeat(1, 1, 3))
        x = self.instance_norm1d(x)
        x = self.instance_norm2d(x.unsqueeze(-1))
        x = self.instance_norm3d(x.unsqueeze(-1))
        return x