from dataclasses import dataclass
from functools import partial
from typing import Any, List, Optional, Tuple
import torch
from torch._C import _disabled_torch_function_impl
from torch.fx.experimental.proxy_tensor import (
from torch.utils import _pytree as pytree
from torch.utils._mode_utils import no_dispatch
from torch.utils._pytree import tree_flatten, tree_map, tree_map_only
def _wrap_comm_result(result: Tuple[Any, Any]) -> Tuple[Any, Any]:

    def wrap(work, e):
        assert isinstance(e, torch.Tensor), 'Excepting collection of tensors as the first element in the return value of communication operations.'
        return _CommResult(e, work)
    work = result[1]
    return (tree_map(partial(wrap, work), result[0]), work)