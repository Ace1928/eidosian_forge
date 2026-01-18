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
class SupportsAnnotations(ExternallyTraversible):
    __slots__ = ()
    _annotations: util.immutabledict[str, Any] = EMPTY_ANNOTATIONS
    proxy_set: util.generic_fn_descriptor[FrozenSet[Any]]
    _is_immutable: bool

    def _annotate(self, values: _AnnotationDict) -> Self:
        raise NotImplementedError()

    @overload
    def _deannotate(self, values: Literal[None]=..., clone: bool=...) -> Self:
        ...

    @overload
    def _deannotate(self, values: Sequence[str]=..., clone: bool=...) -> SupportsAnnotations:
        ...

    def _deannotate(self, values: Optional[Sequence[str]]=None, clone: bool=False) -> SupportsAnnotations:
        raise NotImplementedError()

    @util.memoized_property
    def _annotations_cache_key(self) -> Tuple[Any, ...]:
        anon_map_ = anon_map()
        return self._gen_annotations_cache_key(anon_map_)

    def _gen_annotations_cache_key(self, anon_map: anon_map) -> Tuple[Any, ...]:
        return ('_annotations', tuple(((key, value._gen_cache_key(anon_map, []) if isinstance(value, HasCacheKey) else value) for key, value in [(key, self._annotations[key]) for key in sorted(self._annotations)])))