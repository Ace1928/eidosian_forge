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
class _AttributeOptions(NamedTuple):
    """define Python-local attribute behavior options common to all
    :class:`.MapperProperty` objects.

    Currently this includes dataclass-generation arguments.

    .. versionadded:: 2.0

    """
    dataclasses_init: Union[_NoArg, bool]
    dataclasses_repr: Union[_NoArg, bool]
    dataclasses_default: Union[_NoArg, Any]
    dataclasses_default_factory: Union[_NoArg, Callable[[], Any]]
    dataclasses_compare: Union[_NoArg, bool]
    dataclasses_kw_only: Union[_NoArg, bool]

    def _as_dataclass_field(self, key: str) -> Any:
        """Return a ``dataclasses.Field`` object given these arguments."""
        kw: Dict[str, Any] = {}
        if self.dataclasses_default_factory is not _NoArg.NO_ARG:
            kw['default_factory'] = self.dataclasses_default_factory
        if self.dataclasses_default is not _NoArg.NO_ARG:
            kw['default'] = self.dataclasses_default
        if self.dataclasses_init is not _NoArg.NO_ARG:
            kw['init'] = self.dataclasses_init
        if self.dataclasses_repr is not _NoArg.NO_ARG:
            kw['repr'] = self.dataclasses_repr
        if self.dataclasses_compare is not _NoArg.NO_ARG:
            kw['compare'] = self.dataclasses_compare
        if self.dataclasses_kw_only is not _NoArg.NO_ARG:
            kw['kw_only'] = self.dataclasses_kw_only
        if 'default' in kw and callable(kw['default']):
            warn_deprecated(f'Callable object passed to the ``default`` parameter for attribute {key!r} in a ORM-mapped Dataclasses context is ambiguous, and this use will raise an error in a future release.  If this callable is intended to produce Core level INSERT default values for an underlying ``Column``, use the ``mapped_column.insert_default`` parameter instead.  To establish this callable as providing a default value for instances of the dataclass itself, use the ``default_factory`` dataclasses parameter.', '2.0')
        if 'init' in kw and (not kw['init']) and ('default' in kw) and (not callable(kw['default'])) and ('default_factory' not in kw):
            default = kw.pop('default')
            kw['default_factory'] = lambda: default
        return dataclasses.field(**kw)

    @classmethod
    def _get_arguments_for_make_dataclass(cls, key: str, annotation: _AnnotationScanType, mapped_container: Optional[Any], elem: _T) -> Union[Tuple[str, _AnnotationScanType], Tuple[str, _AnnotationScanType, dataclasses.Field[Any]]]:
        """given attribute key, annotation, and value from a class, return
        the argument tuple we would pass to dataclasses.make_dataclass()
        for this attribute.

        """
        if isinstance(elem, _DCAttributeOptions):
            dc_field = elem._attribute_options._as_dataclass_field(key)
            return (key, annotation, dc_field)
        elif elem is not _NoArg.NO_ARG:
            return (key, annotation, elem)
        elif mapped_container is not None:
            assert False, 'Mapped[] received without a mapping declaration'
        else:
            return (key, annotation)