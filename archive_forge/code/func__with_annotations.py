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
def _with_annotations(self, values: _AnnotationDict) -> Self:
    clone = self.__class__.__new__(self.__class__)
    clone.__dict__ = self.__dict__.copy()
    clone.__dict__.pop('_annotations_cache_key', None)
    clone.__dict__.pop('_generate_cache_key', None)
    clone._annotations = util.immutabledict(values)
    return clone