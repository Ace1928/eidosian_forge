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
def _new_annotation_type(cls: Type[SupportsWrappingAnnotations], base_cls: Type[Annotated]) -> Type[Annotated]:
    """Generates a new class that subclasses Annotated and proxies a given
    element type.

    """
    if issubclass(cls, Annotated):
        return cls
    elif cls in annotated_classes:
        return annotated_classes[cls]
    for super_ in cls.__mro__:
        if super_ in annotated_classes:
            base_cls = annotated_classes[super_]
            break
    annotated_classes[cls] = anno_cls = cast(Type[Annotated], type('Annotated%s' % cls.__name__, (base_cls, cls), {}))
    globals()['Annotated%s' % cls.__name__] = anno_cls
    if '_traverse_internals' in cls.__dict__:
        anno_cls._traverse_internals = list(cls._traverse_internals) + [('_annotations', InternalTraversal.dp_annotations_key)]
    elif cls.__dict__.get('inherit_cache', False):
        anno_cls._traverse_internals = list(cls._traverse_internals) + [('_annotations', InternalTraversal.dp_annotations_key)]
    if cls.__dict__.get('inherit_cache', False):
        anno_cls.inherit_cache = True
    elif 'inherit_cache' in cls.__dict__:
        anno_cls.inherit_cache = cls.__dict__['inherit_cache']
    anno_cls._is_column_operators = issubclass(cls, operators.ColumnOperators)
    return anno_cls