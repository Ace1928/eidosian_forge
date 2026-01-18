from __future__ import annotations
import collections
import dataclasses
import re
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import attributes
from . import clsregistry
from . import exc as orm_exc
from . import instrumentation
from . import mapperlib
from ._typing import _O
from ._typing import attr_is_internal_proxy
from .attributes import InstrumentedAttribute
from .attributes import QueryableAttribute
from .base import _is_mapped_class
from .base import InspectionAttr
from .descriptor_props import CompositeProperty
from .descriptor_props import SynonymProperty
from .interfaces import _AttributeOptions
from .interfaces import _DCAttributeOptions
from .interfaces import _IntrospectsAnnotations
from .interfaces import _MappedAttribute
from .interfaces import _MapsColumns
from .interfaces import MapperProperty
from .mapper import Mapper
from .properties import ColumnProperty
from .properties import MappedColumn
from .util import _extract_mapped_subtype
from .util import _is_mapped_annotation
from .util import class_mapper
from .util import de_stringify_annotation
from .. import event
from .. import exc
from .. import util
from ..sql import expression
from ..sql.base import _NoArg
from ..sql.schema import Column
from ..sql.schema import Table
from ..util import topological
from ..util.typing import _AnnotationScanType
from ..util.typing import is_fwd_ref
from ..util.typing import is_literal
from ..util.typing import Protocol
from ..util.typing import TypedDict
from ..util.typing import typing_get_args
class _DeferredMapperConfig(_ClassScanMapperConfig):
    _cls: weakref.ref[Type[Any]]
    is_deferred = True
    _configs: util.OrderedDict[weakref.ref[Type[Any]], _DeferredMapperConfig] = util.OrderedDict()

    def _early_mapping(self, mapper_kw: _MapperKwArgs) -> None:
        pass

    @property
    def cls(self) -> Type[Any]:
        return self._cls()

    @cls.setter
    def cls(self, class_: Type[Any]) -> None:
        self._cls = weakref.ref(class_, self._remove_config_cls)
        self._configs[self._cls] = self

    @classmethod
    def _remove_config_cls(cls, ref: weakref.ref[Type[Any]]) -> None:
        cls._configs.pop(ref, None)

    @classmethod
    def has_cls(cls, class_: Type[Any]) -> bool:
        return isinstance(class_, type) and weakref.ref(class_) in cls._configs

    @classmethod
    def raise_unmapped_for_cls(cls, class_: Type[Any]) -> NoReturn:
        if hasattr(class_, '_sa_raise_deferred_config'):
            class_._sa_raise_deferred_config()
        raise orm_exc.UnmappedClassError(class_, msg=f'Class {orm_exc._safe_cls_name(class_)} has a deferred mapping on it.  It is not yet usable as a mapped class.')

    @classmethod
    def config_for_cls(cls, class_: Type[Any]) -> _DeferredMapperConfig:
        return cls._configs[weakref.ref(class_)]

    @classmethod
    def classes_for_base(cls, base_cls: Type[Any], sort: bool=True) -> List[_DeferredMapperConfig]:
        classes_for_base = [m for m, cls_ in [(m, m.cls) for m in cls._configs.values()] if cls_ is not None and issubclass(cls_, base_cls)]
        if not sort:
            return classes_for_base
        all_m_by_cls = {m.cls: m for m in classes_for_base}
        tuples: List[Tuple[_DeferredMapperConfig, _DeferredMapperConfig]] = []
        for m_cls in all_m_by_cls:
            tuples.extend(((all_m_by_cls[base_cls], all_m_by_cls[m_cls]) for base_cls in m_cls.__bases__ if base_cls in all_m_by_cls))
        return list(topological.sort(tuples, classes_for_base))

    def map(self, mapper_kw: _MapperKwArgs=util.EMPTY_DICT) -> Mapper[Any]:
        self._configs.pop(self._cls, None)
        return super().map(mapper_kw)