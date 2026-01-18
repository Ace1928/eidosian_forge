import logging
import os
import tempfile
from enum import Enum
from typing import Callable, cast, Dict, Iterable, List, Set
import torch.fx as fx
from torch.fx.passes.shape_prop import TensorMetadata
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_unflatten
class CommType(str, Enum):
    ALLREDUCE = 'allreduce_'
    ALLGATHER = 'allgather_'
    BROADCAST = 'broadcast_'
    REDUCESCATTER = 'reduce_scatter_'
    SCATTER = 'scatter_'