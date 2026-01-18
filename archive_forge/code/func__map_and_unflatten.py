from typing import Any, List, Tuple
from torch.utils._pytree import SUPPORTED_NODES, LeafSpec, PyTree, TreeSpec, _get_node_type, tree_unflatten
def _map_and_unflatten(fn: Any, values: List[Any], spec: TreeSpec) -> PyTree:
    """Utility function to apply a function and unflatten it."""
    return tree_unflatten([fn(i) for i in values], spec)