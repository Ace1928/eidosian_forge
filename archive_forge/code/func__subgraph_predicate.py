import logging
import operator
from typing import Callable, List, Optional, Set, Tuple
from functorch import make_fx
import torch
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.decomposition import select_decomp_table
def _subgraph_predicate(nodes: List[torch.fx.Node]) -> bool:
    num_aten_ops = len([n for n in nodes if str(n.target).startswith('aten.')])
    return num_aten_ops >= MIN_ATEN_OPS_TO_LOWER