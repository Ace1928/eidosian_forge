import collections
import itertools
import logging
import operator
import tempfile
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import (
import torch
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.distributed._spmd.graph_utils import (
from torch.distributed._spmd.iter_graph_module import IterGraphModule
from torch.fx.passes.shape_prop import TensorMetadata
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_unflatten
def _create_meta_tensor_meta(fake_tensor_mode: FakeTensorMode, val: FakeTensor) -> TensorMetadata:
    return TensorMetadata(shape=val.shape, dtype=val.dtype, requires_grad=val.requires_grad, stride=val.stride, memory_format=None, is_quantized=False, qparams={})