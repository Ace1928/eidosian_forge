from typing import Any, Dict, Iterable, List, Tuple
from ._compatibility import compatibility
from torch.utils._pytree import Context, register_pytree_node
def _create_immutable_container(base, mutable_functions):
    container = type('immutable_' + base.__name__, (base,), {})
    for attr in mutable_functions:
        setattr(container, attr, _no_mutation)
    return container