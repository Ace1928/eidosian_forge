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
def _setup_inheriting_columns(self, mapper_kw: _MapperKwArgs) -> None:
    table = self.local_table
    cls = self.cls
    table_args = self.table_args
    declared_columns = self.declared_columns
    if table is None and self.inherits is None and (not _get_immediate_cls_attr(cls, '__no_table__')):
        raise exc.InvalidRequestError('Class %r does not have a __table__ or __tablename__ specified and does not inherit from an existing table-mapped class.' % cls)
    elif self.inherits:
        inherited_mapper_or_config = _declared_mapping_info(self.inherits)
        assert inherited_mapper_or_config is not None
        inherited_table = inherited_mapper_or_config.local_table
        inherited_persist_selectable = inherited_mapper_or_config.persist_selectable
        if table is None:
            if table_args:
                raise exc.ArgumentError("Can't place __table_args__ on an inherited class with no table.")
            if declared_columns and (not isinstance(inherited_table, Table)):
                raise exc.ArgumentError(f"Can't declare columns on single-table-inherited subclass {self.cls}; superclass {self.inherits} is not mapped to a Table")
            for col in declared_columns:
                assert inherited_table is not None
                if col.name in inherited_table.c:
                    if inherited_table.c[col.name] is col:
                        continue
                    raise exc.ArgumentError(f"Column '{col}' on class {cls.__name__} conflicts with existing column '{inherited_table.c[col.name]}'.  If using Declarative, consider using the use_existing_column parameter of mapped_column() to resolve conflicts.")
                if col.primary_key:
                    raise exc.ArgumentError("Can't place primary key columns on an inherited class with no table.")
                if TYPE_CHECKING:
                    assert isinstance(inherited_table, Table)
                inherited_table.append_column(col)
                if inherited_persist_selectable is not None and inherited_persist_selectable is not inherited_table:
                    inherited_persist_selectable._refresh_for_new_column(col)