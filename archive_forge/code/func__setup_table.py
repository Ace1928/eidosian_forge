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
def _setup_table(self, table: Optional[FromClause]=None) -> None:
    cls = self.cls
    cls_as_Decl = cast('MappedClassProtocol[Any]', cls)
    tablename = self.tablename
    table_args = self.table_args
    clsdict_view = self.clsdict_view
    declared_columns = self.declared_columns
    column_ordering = self.column_ordering
    manager = attributes.manager_of_class(cls)
    if '__table__' not in clsdict_view and table is None:
        if hasattr(cls, '__table_cls__'):
            table_cls = cast(Type[Table], util.unbound_method_to_callable(cls.__table_cls__))
        else:
            table_cls = Table
        if tablename is not None:
            args: Tuple[Any, ...] = ()
            table_kw: Dict[str, Any] = {}
            if table_args:
                if isinstance(table_args, dict):
                    table_kw = table_args
                elif isinstance(table_args, tuple):
                    if isinstance(table_args[-1], dict):
                        args, table_kw = (table_args[0:-1], table_args[-1])
                    else:
                        args = table_args
            autoload_with = clsdict_view.get('__autoload_with__')
            if autoload_with:
                table_kw['autoload_with'] = autoload_with
            autoload = clsdict_view.get('__autoload__')
            if autoload:
                table_kw['autoload'] = True
            sorted_columns = sorted(declared_columns, key=lambda c: column_ordering.get(c, 0))
            table = self.set_cls_attribute('__table__', table_cls(tablename, self._metadata_for_cls(manager), *sorted_columns, *args, **table_kw))
    else:
        if table is None:
            table = cls_as_Decl.__table__
        if declared_columns:
            for c in declared_columns:
                if not table.c.contains_column(c):
                    raise exc.ArgumentError("Can't add additional column %r when specifying __table__" % c.key)
    self.local_table = table