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
def _get_full_batch_arg(arg: torch.Tensor) -> torch.Tensor:
    fake_arg = fake_mode.from_tensor(arg)
    arg_dims = [1] * arg.ndim
    arg_dims[data_parallel_mode.input_batch_dim] *= dist.get_world_size()
    return fake_arg.repeat(arg_dims)