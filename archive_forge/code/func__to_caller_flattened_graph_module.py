from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from copy import copy
from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, Callable, cast, Dict, List, Optional, Set, Tuple, Union
from functorch import make_fx
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives
import torch.nn as nn
import torch.utils._pytree as pytree
from torch import fx
from torch._decomp.decompositions import native_layer_norm_backward
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._spmd.data_parallel import gradients_tagging
from torch.distributed._spmd.parallel_mode import (
from torch.distributed._tensor import Placement
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo, CodeGen
from torch.nn.utils import stateless
from torch.nn.utils._named_member_accessor import NamedMemberAccessor
def _to_caller_flattened_graph_module(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Move the responsibility of flattening the input arguments from the graph module to the caller.

    Example:

        output = gm(my_struct)

        gm = gm(to_caller_flattened_graph_module)

        output = gm(*pytree.flatten(my_struct)[0])

    """
    gm._graph._codegen = _PyTreeCodeGenOutputsOnly(pytree_info=_PyTreeInfo(orig_args=None, in_spec=None, out_spec=gm._graph._codegen.pytree_info.out_spec))
    gm.recompile()
    return gm