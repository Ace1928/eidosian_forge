from __future__ import annotations
import collections
import dataclasses
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import Generic
from typing import Iterator
from typing import List
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import exc as orm_exc
from . import path_registry
from .base import _MappedAttribute as _MappedAttribute
from .base import EXT_CONTINUE as EXT_CONTINUE  # noqa: F401
from .base import EXT_SKIP as EXT_SKIP  # noqa: F401
from .base import EXT_STOP as EXT_STOP  # noqa: F401
from .base import InspectionAttr as InspectionAttr  # noqa: F401
from .base import InspectionAttrInfo as InspectionAttrInfo
from .base import MANYTOMANY as MANYTOMANY  # noqa: F401
from .base import MANYTOONE as MANYTOONE  # noqa: F401
from .base import NO_KEY as NO_KEY  # noqa: F401
from .base import NO_VALUE as NO_VALUE  # noqa: F401
from .base import NotExtension as NotExtension  # noqa: F401
from .base import ONETOMANY as ONETOMANY  # noqa: F401
from .base import RelationshipDirection as RelationshipDirection  # noqa: F401
from .base import SQLORMOperations
from .. import ColumnElement
from .. import exc as sa_exc
from .. import inspection
from .. import util
from ..sql import operators
from ..sql import roles
from ..sql import visitors
from ..sql.base import _NoArg
from ..sql.base import ExecutableOption
from ..sql.cache_key import HasCacheKey
from ..sql.operators import ColumnOperators
from ..sql.schema import Column
from ..sql.type_api import TypeEngine
from ..util import warn_deprecated
from ..util.typing import RODescriptorReference
from ..util.typing import TypedDict
class _IntrospectsAnnotations:
    __slots__ = ()

    @classmethod
    def _mapper_property_name(cls) -> str:
        return cls.__name__

    def found_in_pep593_annotated(self) -> Any:
        """return a copy of this object to use in declarative when the
        object is found inside of an Annotated object."""
        raise NotImplementedError(f'Use of the {self._mapper_property_name()!r} construct inside of an Annotated object is not yet supported.')

    def declarative_scan(self, decl_scan: _ClassScanMapperConfig, registry: RegistryType, cls: Type[Any], originating_module: Optional[str], key: str, mapped_container: Optional[Type[Mapped[Any]]], annotation: Optional[_AnnotationScanType], extracted_mapped_annotation: Optional[_AnnotationScanType], is_dataclass_field: bool) -> None:
        """Perform class-specific initializaton at early declarative scanning
        time.

        .. versionadded:: 2.0

        """

    def _raise_for_required(self, key: str, cls: Type[Any]) -> NoReturn:
        raise sa_exc.ArgumentError(f'Python typing annotation is required for attribute "{cls.__name__}.{key}" when primary argument(s) for "{self._mapper_property_name()}" construct are None or not present')