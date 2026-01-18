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
class _RawColumnEntity(_ColumnEntity):
    entity_zero = None
    mapper = None
    supports_single_entity = False
    __slots__ = ('expr', 'column', '_label_name', 'entity_zero_or_selectable', '_extra_entities')

    def __init__(self, compile_state, column, entities_collection, raw_column_index, is_current_entities, parent_bundle=None):
        self.expr = column
        self.raw_column_index = raw_column_index
        self.translate_raw_column = raw_column_index is not None
        if column._is_star:
            compile_state.compile_options += {'_is_star': True}
        if not is_current_entities or column._is_text_clause:
            self._label_name = None
        else:
            self._label_name = compile_state._label_convention(column)
        if parent_bundle:
            parent_bundle._entities.append(self)
        else:
            entities_collection.append(self)
        self.column = column
        self.entity_zero_or_selectable = self.column._from_objects[0] if self.column._from_objects else None
        self._extra_entities = (self.expr, self.column)
        self._fetch_column = self._row_processor = None

    def corresponds_to(self, entity):
        return False

    def setup_dml_returning_compile_state(self, compile_state: ORMCompileState, adapter: DMLReturningColFilter) -> None:
        return self.setup_compile_state(compile_state)

    def setup_compile_state(self, compile_state):
        current_adapter = compile_state._get_current_adapter()
        if current_adapter:
            column = current_adapter(self.column, False)
            if column is None:
                return
        else:
            column = self.column
        if column._annotations:
            column = column._deannotate()
        compile_state.dedupe_columns.add(column)
        compile_state.primary_columns.append(column)
        self._fetch_column = column