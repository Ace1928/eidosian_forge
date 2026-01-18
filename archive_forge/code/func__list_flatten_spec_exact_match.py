from collections import namedtuple
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Type, Optional
from torch.utils._pytree import LeafSpec, PyTree, TreeSpec
def _list_flatten_spec_exact_match(d: List[Any], spec: TreeSpec) -> bool:
    return len(d) == len(spec.children_specs)