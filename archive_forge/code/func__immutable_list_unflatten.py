from typing import Any, Dict, Iterable, List, Tuple
from ._compatibility import compatibility
from torch.utils._pytree import Context, register_pytree_node
def _immutable_list_unflatten(values: Iterable[Any], context: Context) -> List[Any]:
    return immutable_list(values)