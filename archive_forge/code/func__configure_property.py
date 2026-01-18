from __future__ import annotations
from collections import deque
from functools import reduce
from itertools import chain
import sys
import threading
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Deque
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import attributes
from . import exc as orm_exc
from . import instrumentation
from . import loading
from . import properties
from . import util as orm_util
from ._typing import _O
from .base import _class_to_mapper
from .base import _parse_mapper_argument
from .base import _state_mapper
from .base import PassiveFlag
from .base import state_str
from .interfaces import _MappedAttribute
from .interfaces import EXT_SKIP
from .interfaces import InspectionAttr
from .interfaces import MapperProperty
from .interfaces import ORMEntityColumnsClauseRole
from .interfaces import ORMFromClauseRole
from .interfaces import StrategizedProperty
from .path_registry import PathRegistry
from .. import event
from .. import exc as sa_exc
from .. import inspection
from .. import log
from .. import schema
from .. import sql
from .. import util
from ..event import dispatcher
from ..event import EventTarget
from ..sql import base as sql_base
from ..sql import coercions
from ..sql import expression
from ..sql import operators
from ..sql import roles
from ..sql import TableClause
from ..sql import util as sql_util
from ..sql import visitors
from ..sql.cache_key import MemoizedHasCacheKey
from ..sql.elements import KeyedColumnElement
from ..sql.schema import Column
from ..sql.schema import Table
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..util import HasMemoized
from ..util import HasMemoized_ro_memoized_attribute
from ..util.typing import Literal
@util.preload_module('sqlalchemy.orm.descriptor_props')
def _configure_property(self, key: str, prop_arg: Union[KeyedColumnElement[Any], MapperProperty[Any]], *, init: bool=True, setparent: bool=True, warn_for_existing: bool=False) -> MapperProperty[Any]:
    descriptor_props = util.preloaded.orm_descriptor_props
    self._log('_configure_property(%s, %s)', key, prop_arg.__class__.__name__)
    if not isinstance(prop_arg, MapperProperty):
        prop: MapperProperty[Any] = self._property_from_column(key, prop_arg)
    else:
        prop = prop_arg
    if isinstance(prop, properties.ColumnProperty):
        col = self.persist_selectable.corresponding_column(prop.columns[0])
        if col is None and self.inherits:
            path = [self]
            for m in self.inherits.iterate_to_root():
                col = m.local_table.corresponding_column(prop.columns[0])
                if col is not None:
                    for m2 in path:
                        m2.persist_selectable._refresh_for_new_column(col)
                    col = self.persist_selectable.corresponding_column(prop.columns[0])
                    break
                path.append(m)
        if col is None:
            col = prop.columns[0]
            if hasattr(self, '_readonly_props') and (not hasattr(col, 'table') or col.table not in self._cols_by_table):
                self._readonly_props.add(prop)
        elif hasattr(self, '_cols_by_table') and col.table in self._cols_by_table and (col not in self._cols_by_table[col.table]):
            self._cols_by_table[col.table].add(col)
        if not hasattr(prop, '_is_polymorphic_discriminator'):
            prop._is_polymorphic_discriminator = col is self.polymorphic_on or prop.columns[0] is self.polymorphic_on
        if isinstance(col, expression.Label):
            col.key = col._tq_key_label = key
        self.columns.add(col, key)
        for col in prop.columns:
            for proxy_col in col.proxy_set:
                self._columntoproperty[proxy_col] = prop
    if getattr(prop, 'key', key) != key:
        util.warn(f'ORM mapped property {self.class_.__name__}.{prop.key} being assigned to attribute {key!r} is already associated with attribute {prop.key!r}. The attribute will be de-associated from {prop.key!r}.')
    prop.key = key
    if setparent:
        prop.set_parent(self, init)
    if key in self._props and getattr(self._props[key], '_mapped_by_synonym', False):
        syn = self._props[key]._mapped_by_synonym
        raise sa_exc.ArgumentError("Can't call map_column=True for synonym %r=%r, a ColumnProperty already exists keyed to the name %r for column %r" % (syn, key, key, syn))
    if key in self._props and (not isinstance(self._props[key], descriptor_props.ConcreteInheritedProperty)) and (not isinstance(prop, descriptor_props.SynonymProperty)):
        if warn_for_existing:
            util.warn_deprecated(f'User-placed attribute {self.class_.__name__}.{key} on {self} is replacing an existing ORM-mapped attribute.  Behavior is not fully defined in this case.  This use is deprecated and will raise an error in a future release', '2.0')
        oldprop = self._props[key]
        self._path_registry.pop(oldprop, None)
    elif warn_for_existing and self.class_.__dict__.get(key, None) is not None and (not isinstance(prop, descriptor_props.SynonymProperty)) and (not isinstance(self._props.get(key, None), descriptor_props.ConcreteInheritedProperty)):
        util.warn_deprecated(f'User-placed attribute {self.class_.__name__}.{key} on {self} is replacing an existing class-bound attribute of the same name.  Behavior is not fully defined in this case.  This use is deprecated and will raise an error in a future release', '2.0')
    self._props[key] = prop
    if not self.non_primary:
        prop.instrument_class(self)
    for mapper in self._inheriting_mappers:
        mapper._adapt_inherited_property(key, prop, init)
    if init:
        prop.init()
        prop.post_instrument_class(self)
    if self.configured:
        self._expire_memoizations()
    return prop