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
class DenseTopMLP(nn.Module):

    def __init__(self, dense_dim, dense_out, embedding_dim, top_out_in, top_out_out) -> None:
        super().__init__()
        self.dense_mlp = nn.Sequential(nn.Linear(dense_dim, dense_out))
        self.top_mlp = nn.Sequential(nn.Linear(dense_out + embedding_dim, top_out_in), nn.Linear(top_out_in, top_out_out))

    def forward(self, sparse_feature: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        dense_feature = self.dense_mlp(dense)
        features = torch.cat([dense_feature] + [sparse_feature], dim=1)
        out = self.top_mlp(features)
        return out