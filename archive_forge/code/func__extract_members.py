import copy
from typing import (
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils._named_member_accessor import NamedMemberAccessor
def _extract_members(mod: nn.Module, named_members: Callable[..., Iterable[Tuple[str, Tensor]]], subclass: Callable[[Tensor], Tensor]) -> Tuple[Tuple[Tensor, ...], Tuple[str, ...], Dict[str, List[str]]]:
    all_named_members = tuple(named_members(remove_duplicate=False))
    unique_named_members = tuple(named_members(remove_duplicate=True))
    names_map = create_names_map(unique_named_members, all_named_members)
    memo = {}
    accessor = NamedMemberAccessor(mod)
    for name, p in all_named_members:
        if p not in memo:
            memo[p] = subclass(torch.empty_like(p, device='meta'))
        replacement = memo[p]
        accessor.set_tensor(name, replacement)
    if len(unique_named_members) == 0:
        names, params = ((), ())
    else:
        names, params = zip(*unique_named_members)
    return (params, names, names_map)