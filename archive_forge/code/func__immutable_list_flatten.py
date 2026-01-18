from typing import Any, Dict, Iterable, List, Tuple
from ._compatibility import compatibility
from torch.utils._pytree import Context, register_pytree_node
def _immutable_list_flatten(d: List[Any]) -> Tuple[List[Any], Context]:
    return (d, None)