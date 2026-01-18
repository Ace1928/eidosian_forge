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
def _deep_annotate(element: _SA, annotations: _AnnotationDict, exclude: Optional[Sequence[SupportsAnnotations]]=None, *, detect_subquery_cols: bool=False, ind_cols_on_fromclause: bool=False, annotate_callable: Optional[Callable[[SupportsAnnotations, _AnnotationDict], SupportsAnnotations]]=None) -> _SA:
    """Deep copy the given ClauseElement, annotating each element
    with the given annotations dictionary.

    Elements within the exclude collection will be cloned but not annotated.

    """
    cloned_ids: Dict[int, SupportsAnnotations] = {}

    def clone(elem: SupportsAnnotations, **kw: Any) -> SupportsAnnotations:
        kw['detect_subquery_cols'] = detect_subquery_cols
        id_ = id(elem)
        if id_ in cloned_ids:
            return cloned_ids[id_]
        if exclude and hasattr(elem, 'proxy_set') and elem.proxy_set.intersection(exclude):
            newelem = elem._clone(clone=clone, **kw)
        elif annotations != elem._annotations:
            if detect_subquery_cols and elem._is_immutable:
                to_annotate = elem._clone(clone=clone, **kw)
            else:
                to_annotate = elem
            if annotate_callable:
                newelem = annotate_callable(to_annotate, annotations)
            else:
                newelem = _safe_annotate(to_annotate, annotations)
        else:
            newelem = elem
        newelem._copy_internals(clone=clone, ind_cols_on_fromclause=ind_cols_on_fromclause)
        cloned_ids[id_] = newelem
        return newelem
    if element is not None:
        element = cast(_SA, clone(element))
    clone = None
    return element