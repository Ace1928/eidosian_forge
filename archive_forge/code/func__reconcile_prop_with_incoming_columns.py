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
def _reconcile_prop_with_incoming_columns(self, key: str, existing_prop: MapperProperty[Any], warn_only: bool, incoming_prop: Optional[ColumnProperty[Any]]=None, single_column: Optional[KeyedColumnElement[Any]]=None) -> ColumnProperty[Any]:
    if incoming_prop and (self.concrete or not isinstance(existing_prop, properties.ColumnProperty)):
        return incoming_prop
    existing_column = existing_prop.columns[0]
    if incoming_prop and existing_column in incoming_prop.columns:
        return incoming_prop
    if incoming_prop is None:
        assert single_column is not None
        incoming_column = single_column
        equated_pair_key = (existing_prop.columns[0], incoming_column)
    else:
        assert single_column is None
        incoming_column = incoming_prop.columns[0]
        equated_pair_key = (incoming_column, existing_prop.columns[0])
    if (not self._inherits_equated_pairs or equated_pair_key not in self._inherits_equated_pairs) and (not existing_column.shares_lineage(incoming_column)) and (existing_column is not self.version_id_col) and (incoming_column is not self.version_id_col):
        msg = "Implicitly combining column %s with column %s under attribute '%s'.  Please configure one or more attributes for these same-named columns explicitly." % (existing_prop.columns[-1], incoming_column, key)
        if warn_only:
            util.warn(msg)
        else:
            raise sa_exc.InvalidRequestError(msg)
    new_prop = existing_prop.copy()
    new_prop.columns.insert(0, incoming_column)
    self._log('inserting column to existing list in properties.ColumnProperty %s', key)
    return new_prop