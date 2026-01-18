import torch
import torch.nn as nn
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from typing import List, Any, Dict, Optional, Union, NamedTuple
from collections import defaultdict
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.hooks import RemovableHandle
from torch._decomp import register_decomposition
from math import prod
from functools import wraps
def _pytreeify_preserve_structure(f):

    @wraps(f)
    def nf(args):
        flat_args, spec = tree_flatten(args)
        out = f(*flat_args)
        return tree_unflatten(out, spec)
    return nf