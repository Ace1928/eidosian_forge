import torch
from torch.fx import Node
from torch.fx._compatibility import compatibility
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
from torch.utils._pytree import tree_map_only
from torch.utils import _pytree as pytree
from torch.multiprocessing.reductions import StorageWeakRef
import _operator
from enum import Enum
import itertools
from typing import Set, Dict
from collections import defaultdict
def _add_to_map(x):
    if isinstance(x, FakeTensor):
        storage_to_nodes[StorageWeakRef(x._typed_storage())].add(n)