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
class ModelMultipleOps(torch.nn.Module):

    def __init__(self):
        super().__init__()
        norm_layer = nn.BatchNorm2d
        inplanes = 3
        self.conv1 = nn.Conv2d(inplanes, inplanes, (1, 1), bias=False)
        self.conv2 = nn.Conv2d(inplanes, inplanes, (1, 1), bias=False)
        self.bn1 = norm_layer(inplanes)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.downsample = torch.nn.Identity()
        self.skip_add = nn.quantized.FloatFunctional()
        self.cat = nn.quantized.FloatFunctional()
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(12, 6)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        identity = self.downsample(x)
        out = self.skip_add.add(out, identity)
        out = self.relu2(out)
        out = self.avgpool(out)
        out = self.conv2(out)
        out = torch.nn.functional.max_pool2d(out, 2, 2)
        out = self.cat.cat([out, out])
        out = out.reshape(-1, 3 * 2 * 2)
        out = self.fc(out)
        return out