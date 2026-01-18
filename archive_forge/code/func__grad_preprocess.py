from typing import List, Tuple
import torch
from torch._vmap_internals import _vmap
from . import forward_ad as fwAD
def _grad_preprocess(inputs, create_graph, need_graph):
    res = []
    for inp in inputs:
        if create_graph and inp.requires_grad:
            if not inp.is_sparse:
                res.append(inp.view_as(inp))
            else:
                res.append(inp.clone())
        else:
            res.append(inp.detach().requires_grad_(need_graph))
    return tuple(res)