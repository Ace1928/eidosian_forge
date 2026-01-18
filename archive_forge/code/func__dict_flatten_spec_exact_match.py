from collections import namedtuple
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Type, Optional
from torch.utils._pytree import LeafSpec, PyTree, TreeSpec
def _dict_flatten_spec_exact_match(d: Dict[Any, Any], spec: TreeSpec) -> bool:
    return len(d) == len(spec.context)