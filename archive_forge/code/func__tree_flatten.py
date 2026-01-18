from typing import Any, List, Tuple
from torch.utils._pytree import SUPPORTED_NODES, LeafSpec, PyTree, TreeSpec, _get_node_type, tree_unflatten
def _tree_flatten(pytree: PyTree) -> Tuple[List[Any], TreeSpec]:
    """Copy of :func:`torch.utils._pytree.tree_flatten` using our custom leaf function."""
    if _is_leaf_or_primitive_container(pytree):
        return ([pytree], LeafSpec())
    node_type = _get_node_type(pytree)
    flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
    child_pytrees, context = flatten_fn(pytree)
    result: List[Any] = []
    children_specs: List['TreeSpec'] = []
    for child in child_pytrees:
        flat, child_spec = _tree_flatten(child)
        result += flat
        children_specs.append(child_spec)
    return (result, TreeSpec(node_type, context, children_specs))