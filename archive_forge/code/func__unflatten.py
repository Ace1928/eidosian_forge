from typing import Any, Callable, List, Optional, Sequence, Tuple, TypeVar, cast
from ..config import registry
from ..model import Model
def _unflatten(flat: FlatT[ItemT], lens: List[int]) -> NestedT[ItemT]:
    nested = []
    for l in lens:
        nested.append(flat[:l])
        flat = flat[l:]
    return nested