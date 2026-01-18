import logging
import operator
from collections import defaultdict
from typing import Set
import torch
from torch.fx import GraphModule
from torch.fx.passes.backends.cudagraphs import partition_cudagraphs
from torch.multiprocessing.reductions import StorageWeakRef
from torch.nn import Module
from torch.utils._pytree import tree_map
from .common import aot_autograd
from .registry import register_backend
This isn't registered as a backend, but is used in some benchmarks