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
def _collect_annotation(self, name: str, raw_annotation: _AnnotationScanType, originating_class: Type[Any], expect_mapped: Optional[bool], attr_value: Any) -> Optional[_CollectedAnnotation]:
    if name in self.collected_annotations:
        return self.collected_annotations[name]
    if raw_annotation is None:
        return None
    is_dataclass = self.is_dataclass_prior_to_mapping
    allow_unmapped = self.allow_unmapped_annotations
    if expect_mapped is None:
        is_dataclass_field = isinstance(attr_value, dataclasses.Field)
        expect_mapped = not is_dataclass_field and (not allow_unmapped) and (attr_value is None or isinstance(attr_value, _MappedAttribute))
    else:
        is_dataclass_field = False
    is_dataclass_field = False
    extracted = _extract_mapped_subtype(raw_annotation, self.cls, originating_class.__module__, name, type(attr_value), required=False, is_dataclass_field=is_dataclass_field, expect_mapped=expect_mapped and (not is_dataclass))
    if extracted is None:
        return None
    extracted_mapped_annotation, mapped_container = extracted
    if attr_value is None and (not is_literal(extracted_mapped_annotation)):
        for elem in typing_get_args(extracted_mapped_annotation):
            if isinstance(elem, str) or is_fwd_ref(elem, check_generic=True):
                elem = de_stringify_annotation(self.cls, elem, originating_class.__module__, include_generic=True)
            if isinstance(elem, _IntrospectsAnnotations):
                attr_value = elem.found_in_pep593_annotated()
    self.collected_annotations[name] = ca = _CollectedAnnotation(raw_annotation, mapped_container, extracted_mapped_annotation, is_dataclass, attr_value, originating_class.__module__, originating_class)
    return ca