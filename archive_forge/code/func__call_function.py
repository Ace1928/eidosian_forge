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
def _call_function(gm: IterGraphModule, fake_tensor_mode: FakeTensorMode, meta_val: Optional[FakeTensor], function: Any, *args: Any, **kwargs: Any) -> fx.Node:
    node = gm.graph.call_function(function, args, kwargs)
    if meta_val is None:
        flat_args, spec = tree_flatten((args, kwargs))
        new_flat_args = []
        memory_format = None
        for arg in flat_args:
            if not isinstance(arg, fx.Node):
                new_flat_args.append(arg)
                continue
            val = arg.meta['val']
            new_flat_args.append(_create_meta_val(fake_tensor_mode, val))
        fake_args, fake_kwargs = tree_unflatten(new_flat_args, spec)
        new_meta_val = function(*fake_args, **fake_kwargs)
    else:
        new_meta_val = meta_val
    node.meta['val'] = new_meta_val
    node.meta['tensor_meta'] = _create_meta_tensor_meta(fake_tensor_mode, new_meta_val)
    return node