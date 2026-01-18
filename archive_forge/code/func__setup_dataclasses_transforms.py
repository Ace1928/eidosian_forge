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
def _setup_dataclasses_transforms(self) -> None:
    dataclass_setup_arguments = self.dataclass_setup_arguments
    if not dataclass_setup_arguments:
        return
    if '__dataclass_fields__' in self.cls.__dict__:
        raise exc.InvalidRequestError(f"Class {self.cls} is already a dataclass; ensure that base classes / decorator styles of establishing dataclasses are not being mixed. This can happen if a class that inherits from 'MappedAsDataclass', even indirectly, is been mapped with '@registry.mapped_as_dataclass'")
    warn_for_non_dc_attrs = collections.defaultdict(list)

    def _allow_dataclass_field(key: str, originating_class: Type[Any]) -> bool:
        if originating_class is not self.cls and '__dataclass_fields__' not in originating_class.__dict__:
            warn_for_non_dc_attrs[originating_class].append(key)
        return True
    manager = instrumentation.manager_of_class(self.cls)
    assert manager is not None
    field_list = [_AttributeOptions._get_arguments_for_make_dataclass(key, anno, mapped_container, self.collected_attributes.get(key, _NoArg.NO_ARG)) for key, anno, mapped_container in ((key, mapped_anno if mapped_anno else raw_anno, mapped_container) for key, (raw_anno, mapped_container, mapped_anno, is_dc, attr_value, originating_module, originating_class) in self.collected_annotations.items() if _allow_dataclass_field(key, originating_class) and (key not in self.collected_attributes or not isinstance(self.collected_attributes[key], QueryableAttribute)))]
    if warn_for_non_dc_attrs:
        for originating_class, non_dc_attrs in warn_for_non_dc_attrs.items():
            util.warn_deprecated(f'When transforming {self.cls} to a dataclass, attribute(s) {', '.join((repr(key) for key in non_dc_attrs))} originates from superclass {originating_class}, which is not a dataclass.  This usage is deprecated and will raise an error in SQLAlchemy 2.1.  When declaring SQLAlchemy Declarative Dataclasses, ensure that all mixin classes and other superclasses which include attributes are also a subclass of MappedAsDataclass.', '2.0', code='dcmx')
    annotations = {}
    defaults = {}
    for item in field_list:
        if len(item) == 2:
            name, tp = item
        elif len(item) == 3:
            name, tp, spec = item
            defaults[name] = spec
        else:
            assert False
        annotations[name] = tp
    for k, v in defaults.items():
        setattr(self.cls, k, v)
    self._apply_dataclasses_to_any_class(dataclass_setup_arguments, self.cls, annotations)