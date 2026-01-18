from collections import namedtuple
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Type, Optional
from torch.utils._pytree import LeafSpec, PyTree, TreeSpec
def _tuple_flatten_spec(d: Tuple[Any], spec: TreeSpec) -> List[Any]:
    return [d[i] for i in range(len(spec.children_specs))]