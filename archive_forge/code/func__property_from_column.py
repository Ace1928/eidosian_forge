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
def _property_from_column(self, key: str, column: KeyedColumnElement[Any]) -> ColumnProperty[Any]:
    """generate/update a :class:`.ColumnProperty` given a
        :class:`_schema.Column` or other SQL expression object."""
    descriptor_props = util.preloaded.orm_descriptor_props
    prop = self._props.get(key)
    if isinstance(prop, properties.ColumnProperty):
        return self._reconcile_prop_with_incoming_columns(key, prop, single_column=column, warn_only=prop.parent is not self)
    elif prop is None or isinstance(prop, descriptor_props.ConcreteInheritedProperty):
        return self._make_prop_from_column(key, column)
    else:
        raise sa_exc.ArgumentError("WARNING: when configuring property '%s' on %s, column '%s' conflicts with property '%r'. To resolve this, map the column to the class under a different name in the 'properties' dictionary.  Or, to remove all awareness of the column entirely (including its availability as a foreign key), use the 'include_properties' or 'exclude_properties' mapper arguments to control specifically which table columns get mapped." % (key, self, column.key, prop))