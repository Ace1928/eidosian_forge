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
class _MapperEntity(_QueryEntity):
    """mapper/class/AliasedClass entity"""
    __slots__ = ('expr', 'mapper', 'entity_zero', 'is_aliased_class', 'path', '_extra_entities', '_label_name', '_with_polymorphic_mappers', 'selectable', '_polymorphic_discriminator')
    expr: _InternalEntityType
    mapper: Mapper[Any]
    entity_zero: _InternalEntityType
    is_aliased_class: bool
    path: PathRegistry
    _label_name: str

    def __init__(self, compile_state, entity, entities_collection, is_current_entities):
        entities_collection.append(self)
        if is_current_entities:
            if compile_state._primary_entity is None:
                compile_state._primary_entity = self
            compile_state._has_mapper_entities = True
            compile_state._has_orm_entities = True
        entity = entity._annotations['parententity']
        entity._post_inspect
        ext_info = self.entity_zero = entity
        entity = ext_info.entity
        self.expr = entity
        self.mapper = mapper = ext_info.mapper
        self._extra_entities = (self.expr,)
        if ext_info.is_aliased_class:
            self._label_name = ext_info.name
        else:
            self._label_name = mapper.class_.__name__
        self.is_aliased_class = ext_info.is_aliased_class
        self.path = ext_info._path_registry
        self.selectable = ext_info.selectable
        self._with_polymorphic_mappers = ext_info.with_polymorphic_mappers
        self._polymorphic_discriminator = ext_info.polymorphic_on
        if mapper._should_select_with_poly_adapter:
            compile_state._create_with_polymorphic_adapter(ext_info, self.selectable)
    supports_single_entity = True
    _non_hashable_value = True
    use_id_for_hash = True

    @property
    def type(self):
        return self.mapper.class_

    @property
    def entity_zero_or_selectable(self):
        return self.entity_zero

    def corresponds_to(self, entity):
        return _entity_corresponds_to(self.entity_zero, entity)

    def _get_entity_clauses(self, compile_state):
        adapter = None
        if not self.is_aliased_class:
            if compile_state._polymorphic_adapters:
                adapter = compile_state._polymorphic_adapters.get(self.mapper, None)
        else:
            adapter = self.entity_zero._adapter
        if adapter:
            if compile_state._from_obj_alias:
                ret = adapter.wrap(compile_state._from_obj_alias)
            else:
                ret = adapter
        else:
            ret = compile_state._from_obj_alias
        return ret

    def row_processor(self, context, result):
        compile_state = context.compile_state
        adapter = self._get_entity_clauses(compile_state)
        if compile_state.compound_eager_adapter and adapter:
            adapter = adapter.wrap(compile_state.compound_eager_adapter)
        elif not adapter:
            adapter = compile_state.compound_eager_adapter
        if compile_state._primary_entity is self:
            only_load_props = compile_state.compile_options._only_load_props
            refresh_state = context.refresh_state
        else:
            only_load_props = refresh_state = None
        _instance = loading._instance_processor(self, self.mapper, context, result, self.path, adapter, only_load_props=only_load_props, refresh_state=refresh_state, polymorphic_discriminator=self._polymorphic_discriminator)
        return (_instance, self._label_name, self._extra_entities)

    def setup_dml_returning_compile_state(self, compile_state: ORMCompileState, adapter: DMLReturningColFilter) -> None:
        loading._setup_entity_query(compile_state, self.mapper, self, self.path, adapter, compile_state.primary_columns, with_polymorphic=self._with_polymorphic_mappers, only_load_props=compile_state.compile_options._only_load_props, polymorphic_discriminator=self._polymorphic_discriminator)

    def setup_compile_state(self, compile_state):
        adapter = self._get_entity_clauses(compile_state)
        single_table_crit = self.mapper._single_table_criterion
        if single_table_crit is not None or ('additional_entity_criteria', self.mapper) in compile_state.global_attributes:
            ext_info = self.entity_zero
            compile_state.extra_criteria_entities[ext_info] = (ext_info, ext_info._adapter if ext_info.is_aliased_class else None)
        loading._setup_entity_query(compile_state, self.mapper, self, self.path, adapter, compile_state.primary_columns, with_polymorphic=self._with_polymorphic_mappers, only_load_props=compile_state.compile_options._only_load_props, polymorphic_discriminator=self._polymorphic_discriminator)
        compile_state._fallback_from_clauses.append(self.selectable)