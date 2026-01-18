from typing import Any, List, Tuple
from torch.utils._pytree import SUPPORTED_NODES, LeafSpec, PyTree, TreeSpec, _get_node_type, tree_unflatten
def _is_leaf_or_primitive_container(pytree: PyTree) -> bool:
    """Customized :func:`torch.utils._pytree._is_leaf` to avoid flattening containers of primitives."""
    is_leaf = _get_node_type(pytree) not in SUPPORTED_NODES
    if is_leaf:
        return True
    node_type = _get_node_type(pytree)
    flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
    child_pytrees, _ = flatten_fn(pytree)
    return all((isinstance(child, (int, float, str)) for child in child_pytrees))