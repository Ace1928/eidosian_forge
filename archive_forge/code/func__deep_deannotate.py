from __future__ import annotations
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from . import operators
from .cache_key import HasCacheKey
from .visitors import anon_map
from .visitors import ExternallyTraversible
from .visitors import InternalTraversal
from .. import util
from ..util.typing import Literal
from ..util.typing import Self
def _deep_deannotate(element: Optional[_SA], values: Optional[Sequence[str]]=None) -> Optional[_SA]:
    """Deep copy the given element, removing annotations."""
    cloned: Dict[Any, SupportsAnnotations] = {}

    def clone(elem: SupportsAnnotations, **kw: Any) -> SupportsAnnotations:
        key: Any
        if values:
            key = id(elem)
        else:
            key = elem
        if key not in cloned:
            newelem = elem._deannotate(values=values, clone=True)
            newelem._copy_internals(clone=clone)
            cloned[key] = newelem
            return newelem
        else:
            return cloned[key]
    if element is not None:
        element = cast(_SA, clone(element))
    clone = None
    return element