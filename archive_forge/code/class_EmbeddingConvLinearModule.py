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
class EmbeddingConvLinearModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(num_embeddings=10, embedding_dim=8)
        self.conv = torch.nn.Conv2d(8, 16, (1, 3))
        self.linear = torch.nn.Linear(16, 8)

    def forward(self, indices):
        embeddings = self.emb(indices)
        embeddings = torch.unsqueeze(embeddings, dim=0)
        embeddings = torch.permute(embeddings, (0, 3, 1, 2))
        conv_out = self.conv(embeddings)
        conv_out = torch.permute(conv_out, (0, 2, 3, 1))
        conv_out = torch.squeeze(conv_out, dim=0)
        return self.linear(conv_out)