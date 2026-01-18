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
class SparseNNModel(nn.Module):
    _NUM_EMBEDDINGS = 10
    _EMBEDDING_DIM = 5
    _DENSE_DIM = 4
    _DENSE_OUTPUT = 2
    _TOP_OUT_IN = 2
    _TOP_OUT_OUT = 2
    _TOP_MLP_DIM = 1

    def __init__(self) -> None:
        super().__init__()
        self.model_sparse = EmbBagWrapper(self._NUM_EMBEDDINGS, self._EMBEDDING_DIM)
        self.dense_top = DenseTopMLP(self._DENSE_DIM, self._DENSE_OUTPUT, self._EMBEDDING_DIM, self._TOP_OUT_IN, self._TOP_OUT_OUT)

    def forward(self, sparse_indices: torch.Tensor, sparse_offsets: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        sparse_feature = self.model_sparse(sparse_indices, sparse_offsets)
        out = self.dense_top(sparse_feature, dense)
        return out