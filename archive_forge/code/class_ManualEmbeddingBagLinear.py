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
class ManualEmbeddingBagLinear(nn.Module):

    def __init__(self):
        super().__init__()
        self.emb = nn.EmbeddingBag(num_embeddings=10, embedding_dim=12, mode='sum')
        self.emb.qconfig = default_embedding_qat_qconfig
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.linear = nn.Linear(12, 1).to(dtype=torch.float)
        self.qconfig = get_default_qat_qconfig('qnnpack')

    def forward(self, input: torch.Tensor, offsets: Optional[torch.Tensor]=None, per_sample_weights: Optional[torch.Tensor]=None):
        x = self.emb(input, offsets, per_sample_weights)
        x = self.quant(x)
        x = self.linear(x)
        return self.dequant(x)