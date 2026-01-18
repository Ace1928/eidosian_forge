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
class _ColumnEntity(_QueryEntity):
    __slots__ = ('_fetch_column', '_row_processor', 'raw_column_index', 'translate_raw_column')

    @classmethod
    def _for_columns(cls, compile_state, columns, entities_collection, raw_column_index, is_current_entities, parent_bundle=None):
        for column in columns:
            annotations = column._annotations
            if 'parententity' in annotations:
                _entity = annotations['parententity']
            else:
                _entity = sql_util.extract_first_column_annotation(column, 'parententity')
            if _entity:
                if 'identity_token' in column._annotations:
                    _IdentityTokenEntity(compile_state, column, entities_collection, _entity, raw_column_index, is_current_entities, parent_bundle=parent_bundle)
                else:
                    _ORMColumnEntity(compile_state, column, entities_collection, _entity, raw_column_index, is_current_entities, parent_bundle=parent_bundle)
            else:
                _RawColumnEntity(compile_state, column, entities_collection, raw_column_index, is_current_entities, parent_bundle=parent_bundle)

    @property
    def type(self):
        return self.column.type

    @property
    def _non_hashable_value(self):
        return not self.column.type.hashable

    @property
    def _null_column_type(self):
        return self.column.type._isnull

    def row_processor(self, context, result):
        compile_state = context.compile_state
        if self._row_processor is not None:
            getter, label_name, extra_entities = self._row_processor
            if self.translate_raw_column:
                extra_entities += (context.query._raw_columns[self.raw_column_index],)
            return (getter, label_name, extra_entities)
        if self._fetch_column is not None:
            column = self._fetch_column
        else:
            column = self.column
            if compile_state._from_obj_alias:
                column = compile_state._from_obj_alias.columns[column]
            if column._annotations:
                column = column._deannotate()
        if compile_state.compound_eager_adapter:
            column = compile_state.compound_eager_adapter.columns[column]
        getter = result._getter(column)
        ret = (getter, self._label_name, self._extra_entities)
        self._row_processor = ret
        if self.translate_raw_column:
            extra_entities = self._extra_entities + (context.query._raw_columns[self.raw_column_index],)
            return (getter, self._label_name, extra_entities)
        else:
            return ret