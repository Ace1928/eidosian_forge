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
def _prepare_annotations(target_hierarchy: Type[SupportsWrappingAnnotations], base_cls: Type[Annotated]) -> None:
    for cls in util.walk_subclasses(target_hierarchy):
        _new_annotation_type(cls, base_cls)