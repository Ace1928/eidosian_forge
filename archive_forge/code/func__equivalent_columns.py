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
@HasMemoized.memoized_attribute
def _equivalent_columns(self) -> _EquivalentColumnMap:
    """Create a map of all equivalent columns, based on
        the determination of column pairs that are equated to
        one another based on inherit condition.  This is designed
        to work with the queries that util.polymorphic_union
        comes up with, which often don't include the columns from
        the base table directly (including the subclass table columns
        only).

        The resulting structure is a dictionary of columns mapped
        to lists of equivalent columns, e.g.::

            {
                tablea.col1:
                    {tableb.col1, tablec.col1},
                tablea.col2:
                    {tabled.col2}
            }

        """
    result: _EquivalentColumnMap = {}

    def visit_binary(binary):
        if binary.operator == operators.eq:
            if binary.left in result:
                result[binary.left].add(binary.right)
            else:
                result[binary.left] = {binary.right}
            if binary.right in result:
                result[binary.right].add(binary.left)
            else:
                result[binary.right] = {binary.left}
    for mapper in self.base_mapper.self_and_descendants:
        if mapper.inherit_condition is not None:
            visitors.traverse(mapper.inherit_condition, {}, {'binary': visit_binary})
    return result