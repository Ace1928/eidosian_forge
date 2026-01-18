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
def _gen_annotations_cache_key(self, anon_map: anon_map) -> Tuple[Any, ...]:
    return ('_annotations', tuple(((key, value._gen_cache_key(anon_map, []) if isinstance(value, HasCacheKey) else value) for key, value in [(key, self._annotations[key]) for key in sorted(self._annotations)])))