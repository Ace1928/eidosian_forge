from dataclasses import dataclass
from functools import partial
from typing import Any, List, Optional, Tuple
import torch
from torch._C import _disabled_torch_function_impl
from torch.fx.experimental.proxy_tensor import (
from torch.utils import _pytree as pytree
from torch.utils._mode_utils import no_dispatch
from torch.utils._pytree import tree_flatten, tree_map, tree_map_only
def _wait_comm(comm_result: _CommResult):
    comm_result._work.wait()
    return comm_result._tensor