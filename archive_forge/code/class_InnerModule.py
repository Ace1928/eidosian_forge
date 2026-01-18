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
class InnerModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(5, 8).to(dtype=torch.float)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(8, 5).to(dtype=torch.float)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        return self.relu2(self.fc2(self.relu1(self.fc1(x))))

    def fuse_modules(self):
        fusable_layers = []
        named_children = list(self.named_children())
        for idx, (current_name, layer) in enumerate(named_children):
            if isinstance(layer, torch.nn.Linear):
                if idx >= len(named_children) - 1:
                    break
                if isinstance(named_children[idx + 1][1], torch.nn.ReLU):
                    fusable_layers.append([current_name, named_children[idx + 1][0]])
        if self.training:
            torch.ao.quantization.fuse_modules_qat(self, fusable_layers, inplace=True)
        else:
            torch.ao.quantization.fuse_modules(self, fusable_layers, inplace=True)