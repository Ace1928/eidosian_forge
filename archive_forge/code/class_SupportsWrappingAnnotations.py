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
class SupportsWrappingAnnotations(SupportsAnnotations):
    __slots__ = ()
    _constructor: Callable[..., SupportsWrappingAnnotations]
    if TYPE_CHECKING:

        @util.ro_non_memoized_property
        def entity_namespace(self) -> _EntityNamespace:
            ...

    def _annotate(self, values: _AnnotationDict) -> Self:
        """return a copy of this ClauseElement with annotations
        updated by the given dictionary.

        """
        return Annotated._as_annotated_instance(self, values)

    def _with_annotations(self, values: _AnnotationDict) -> Self:
        """return a copy of this ClauseElement with annotations
        replaced by the given dictionary.

        """
        return Annotated._as_annotated_instance(self, values)

    @overload
    def _deannotate(self, values: Literal[None]=..., clone: bool=...) -> Self:
        ...

    @overload
    def _deannotate(self, values: Sequence[str]=..., clone: bool=...) -> SupportsAnnotations:
        ...

    def _deannotate(self, values: Optional[Sequence[str]]=None, clone: bool=False) -> SupportsAnnotations:
        """return a copy of this :class:`_expression.ClauseElement`
        with annotations
        removed.

        :param values: optional tuple of individual values
         to remove.

        """
        if clone:
            s = self._clone()
            return s
        else:
            return self