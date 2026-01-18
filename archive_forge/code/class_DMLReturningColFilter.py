from __future__ import annotations
import itertools
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import interfaces
from . import loading
from .base import _is_aliased_class
from .interfaces import ORMColumnDescription
from .interfaces import ORMColumnsClauseRole
from .path_registry import PathRegistry
from .util import _entity_corresponds_to
from .util import _ORMJoin
from .util import _TraceAdaptRole
from .util import AliasedClass
from .util import Bundle
from .util import ORMAdapter
from .util import ORMStatementAdapter
from .. import exc as sa_exc
from .. import future
from .. import inspect
from .. import sql
from .. import util
from ..sql import coercions
from ..sql import expression
from ..sql import roles
from ..sql import util as sql_util
from ..sql import visitors
from ..sql._typing import _TP
from ..sql._typing import is_dml
from ..sql._typing import is_insert_update
from ..sql._typing import is_select_base
from ..sql.base import _select_iterables
from ..sql.base import CacheableOptions
from ..sql.base import CompileState
from ..sql.base import Executable
from ..sql.base import Generative
from ..sql.base import Options
from ..sql.dml import UpdateBase
from ..sql.elements import GroupedElement
from ..sql.elements import TextClause
from ..sql.selectable import CompoundSelectState
from ..sql.selectable import LABEL_STYLE_DISAMBIGUATE_ONLY
from ..sql.selectable import LABEL_STYLE_NONE
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..sql.selectable import Select
from ..sql.selectable import SelectLabelStyle
from ..sql.selectable import SelectState
from ..sql.selectable import TypedReturnsRows
from ..sql.visitors import InternalTraversal
class DMLReturningColFilter:
    """an adapter used for the DML RETURNING case.

    Has a subset of the interface used by
    :class:`.ORMAdapter` and is used for :class:`._QueryEntity`
    instances to set up their columns as used in RETURNING for a
    DML statement.

    """
    __slots__ = ('mapper', 'columns', '__weakref__')

    def __init__(self, target_mapper, immediate_dml_mapper):
        if immediate_dml_mapper is not None and target_mapper.local_table is not immediate_dml_mapper.local_table:
            self.mapper = immediate_dml_mapper
        else:
            self.mapper = target_mapper
        self.columns = self.columns = util.WeakPopulateDict(self.adapt_check_present)

    def __call__(self, col, as_filter):
        for cc in sql_util._find_columns(col):
            c2 = self.adapt_check_present(cc)
            if c2 is not None:
                return col
        else:
            return None

    def adapt_check_present(self, col):
        mapper = self.mapper
        prop = mapper._columntoproperty.get(col, None)
        if prop is None:
            return None
        return mapper.local_table.c.corresponding_column(col)