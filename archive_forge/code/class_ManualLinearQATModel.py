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
class ManualLinearQATModel(torch.nn.Module):
    """A Module with manually inserted `QuantStub` and `DeQuantStub`
    """

    def __init__(self, qengine):
        super().__init__()
        self.qconfig = torch.ao.quantization.get_default_qat_qconfig(qengine)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.fc1 = torch.nn.Linear(5, 1).to(dtype=torch.float)
        self.fc2 = torch.nn.Linear(1, 10).to(dtype=torch.float)

    def forward(self, x):
        x = self.quant(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.dequant(x)