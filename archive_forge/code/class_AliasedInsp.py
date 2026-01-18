from __future__ import annotations
import enum
import functools
import re
import types
import typing
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Match
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import attributes  # noqa
from . import exc
from ._typing import _O
from ._typing import insp_is_aliased_class
from ._typing import insp_is_mapper
from ._typing import prop_is_relationship
from .base import _class_to_mapper as _class_to_mapper
from .base import _MappedAnnotationBase
from .base import _never_set as _never_set  # noqa: F401
from .base import _none_set as _none_set  # noqa: F401
from .base import attribute_str as attribute_str  # noqa: F401
from .base import class_mapper as class_mapper
from .base import DynamicMapped
from .base import InspectionAttr as InspectionAttr
from .base import instance_str as instance_str  # noqa: F401
from .base import Mapped
from .base import object_mapper as object_mapper
from .base import object_state as object_state  # noqa: F401
from .base import opt_manager_of_class
from .base import ORMDescriptor
from .base import state_attribute_str as state_attribute_str  # noqa: F401
from .base import state_class_str as state_class_str  # noqa: F401
from .base import state_str as state_str  # noqa: F401
from .base import WriteOnlyMapped
from .interfaces import CriteriaOption
from .interfaces import MapperProperty as MapperProperty
from .interfaces import ORMColumnsClauseRole
from .interfaces import ORMEntityColumnsClauseRole
from .interfaces import ORMFromClauseRole
from .path_registry import PathRegistry as PathRegistry
from .. import event
from .. import exc as sa_exc
from .. import inspection
from .. import sql
from .. import util
from ..engine.result import result_tuple
from ..sql import coercions
from ..sql import expression
from ..sql import lambdas
from ..sql import roles
from ..sql import util as sql_util
from ..sql import visitors
from ..sql._typing import is_selectable
from ..sql.annotation import SupportsCloneAnnotations
from ..sql.base import ColumnCollection
from ..sql.cache_key import HasCacheKey
from ..sql.cache_key import MemoizedHasCacheKey
from ..sql.elements import ColumnElement
from ..sql.elements import KeyedColumnElement
from ..sql.selectable import FromClause
from ..util.langhelpers import MemoizedSlots
from ..util.typing import de_stringify_annotation as _de_stringify_annotation
from ..util.typing import (
from ..util.typing import eval_name_only as _eval_name_only
from ..util.typing import is_origin_of_cls
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import typing_get_origin
@inspection._self_inspects
class AliasedInsp(ORMEntityColumnsClauseRole[_O], ORMFromClauseRole, HasCacheKey, InspectionAttr, MemoizedSlots, inspection.Inspectable['AliasedInsp[_O]'], Generic[_O]):
    """Provide an inspection interface for an
    :class:`.AliasedClass` object.

    The :class:`.AliasedInsp` object is returned
    given an :class:`.AliasedClass` using the
    :func:`_sa.inspect` function::

        from sqlalchemy import inspect
        from sqlalchemy.orm import aliased

        my_alias = aliased(MyMappedClass)
        insp = inspect(my_alias)

    Attributes on :class:`.AliasedInsp`
    include:

    * ``entity`` - the :class:`.AliasedClass` represented.
    * ``mapper`` - the :class:`_orm.Mapper` mapping the underlying class.
    * ``selectable`` - the :class:`_expression.Alias`
      construct which ultimately
      represents an aliased :class:`_schema.Table` or
      :class:`_expression.Select`
      construct.
    * ``name`` - the name of the alias.  Also is used as the attribute
      name when returned in a result tuple from :class:`_query.Query`.
    * ``with_polymorphic_mappers`` - collection of :class:`_orm.Mapper`
      objects
      indicating all those mappers expressed in the select construct
      for the :class:`.AliasedClass`.
    * ``polymorphic_on`` - an alternate column or SQL expression which
      will be used as the "discriminator" for a polymorphic load.

    .. seealso::

        :ref:`inspection_toplevel`

    """
    __slots__ = ('__weakref__', '_weak_entity', 'mapper', 'selectable', 'name', '_adapt_on_names', 'with_polymorphic_mappers', 'polymorphic_on', '_use_mapper_path', '_base_alias', 'represents_outer_join', 'persist_selectable', 'local_table', '_is_with_polymorphic', '_with_polymorphic_entities', '_adapter', '_target', '__clause_element__', '_memoized_values', '_all_column_expressions', '_nest_adapters')
    _cache_key_traversal = [('name', visitors.ExtendedInternalTraversal.dp_string), ('_adapt_on_names', visitors.ExtendedInternalTraversal.dp_boolean), ('_use_mapper_path', visitors.ExtendedInternalTraversal.dp_boolean), ('_target', visitors.ExtendedInternalTraversal.dp_inspectable), ('selectable', visitors.ExtendedInternalTraversal.dp_clauseelement), ('with_polymorphic_mappers', visitors.InternalTraversal.dp_has_cache_key_list), ('polymorphic_on', visitors.InternalTraversal.dp_clauseelement)]
    mapper: Mapper[_O]
    selectable: FromClause
    _adapter: ORMAdapter
    with_polymorphic_mappers: Sequence[Mapper[Any]]
    _with_polymorphic_entities: Sequence[AliasedInsp[Any]]
    _weak_entity: weakref.ref[AliasedClass[_O]]
    'the AliasedClass that refers to this AliasedInsp'
    _target: Union[Type[_O], AliasedClass[_O]]
    'the thing referenced by the AliasedClass/AliasedInsp.\n\n    In the vast majority of cases, this is the mapped class.  However\n    it may also be another AliasedClass (alias of alias).\n\n    '

    def __init__(self, entity: AliasedClass[_O], inspected: _InternalEntityType[_O], selectable: FromClause, name: Optional[str], with_polymorphic_mappers: Optional[Sequence[Mapper[Any]]], polymorphic_on: Optional[ColumnElement[Any]], _base_alias: Optional[AliasedInsp[Any]], _use_mapper_path: bool, adapt_on_names: bool, represents_outer_join: bool, nest_adapters: bool):
        mapped_class_or_ac = inspected.entity
        mapper = inspected.mapper
        self._weak_entity = weakref.ref(entity)
        self.mapper = mapper
        self.selectable = self.persist_selectable = self.local_table = selectable
        self.name = name
        self.polymorphic_on = polymorphic_on
        self._base_alias = weakref.ref(_base_alias or self)
        self._use_mapper_path = _use_mapper_path
        self.represents_outer_join = represents_outer_join
        self._nest_adapters = nest_adapters
        if with_polymorphic_mappers:
            self._is_with_polymorphic = True
            self.with_polymorphic_mappers = with_polymorphic_mappers
            self._with_polymorphic_entities = []
            for poly in self.with_polymorphic_mappers:
                if poly is not mapper:
                    ent = AliasedClass(poly.class_, selectable, base_alias=self, adapt_on_names=adapt_on_names, use_mapper_path=_use_mapper_path)
                    setattr(self.entity, poly.class_.__name__, ent)
                    self._with_polymorphic_entities.append(ent._aliased_insp)
        else:
            self._is_with_polymorphic = False
            self.with_polymorphic_mappers = [mapper]
        self._adapter = ORMAdapter(_TraceAdaptRole.ALIASED_INSP, mapper, selectable=selectable, equivalents=mapper._equivalent_columns, adapt_on_names=adapt_on_names, anonymize_labels=True, adapt_from_selectables={m.selectable for m in self.with_polymorphic_mappers if not adapt_on_names}, limit_on_entity=False)
        if nest_adapters:
            assert isinstance(inspected, AliasedInsp)
            self._adapter = inspected._adapter.wrap(self._adapter)
        self._adapt_on_names = adapt_on_names
        self._target = mapped_class_or_ac

    @classmethod
    def _alias_factory(cls, element: Union[_EntityType[_O], FromClause], alias: Optional[FromClause]=None, name: Optional[str]=None, flat: bool=False, adapt_on_names: bool=False) -> Union[AliasedClass[_O], FromClause]:
        if isinstance(element, FromClause):
            if adapt_on_names:
                raise sa_exc.ArgumentError('adapt_on_names only applies to ORM elements')
            if name:
                return element.alias(name=name, flat=flat)
            else:
                return coercions.expect(roles.AnonymizedFromClauseRole, element, flat=flat)
        else:
            return AliasedClass(element, alias=alias, flat=flat, name=name, adapt_on_names=adapt_on_names)

    @classmethod
    def _with_polymorphic_factory(cls, base: Union[Type[_O], Mapper[_O]], classes: Union[Literal['*'], Iterable[_EntityType[Any]]], selectable: Union[Literal[False, None], FromClause]=False, flat: bool=False, polymorphic_on: Optional[ColumnElement[Any]]=None, aliased: bool=False, innerjoin: bool=False, adapt_on_names: bool=False, _use_mapper_path: bool=False) -> AliasedClass[_O]:
        primary_mapper = _class_to_mapper(base)
        if selectable not in (None, False) and flat:
            raise sa_exc.ArgumentError("the 'flat' and 'selectable' arguments cannot be passed simultaneously to with_polymorphic()")
        mappers, selectable = primary_mapper._with_polymorphic_args(classes, selectable, innerjoin=innerjoin)
        if aliased or flat:
            assert selectable is not None
            selectable = selectable._anonymous_fromclause(flat=flat)
        return AliasedClass(base, selectable, with_polymorphic_mappers=mappers, adapt_on_names=adapt_on_names, with_polymorphic_discriminator=polymorphic_on, use_mapper_path=_use_mapper_path, represents_outer_join=not innerjoin)

    @property
    def entity(self) -> AliasedClass[_O]:
        ent = self._weak_entity()
        if ent is None:
            ent = AliasedClass._reconstitute_from_aliased_insp(self)
            self._weak_entity = weakref.ref(ent)
        return ent
    is_aliased_class = True
    'always returns True'

    def _memoized_method___clause_element__(self) -> FromClause:
        return self.selectable._annotate({'parentmapper': self.mapper, 'parententity': self, 'entity_namespace': self})._set_propagate_attrs({'compile_state_plugin': 'orm', 'plugin_subject': self})

    @property
    def entity_namespace(self) -> AliasedClass[_O]:
        return self.entity

    @property
    def class_(self) -> Type[_O]:
        """Return the mapped class ultimately represented by this
        :class:`.AliasedInsp`."""
        return self.mapper.class_

    @property
    def _path_registry(self) -> AbstractEntityRegistry:
        if self._use_mapper_path:
            return self.mapper._path_registry
        else:
            return PathRegistry.per_mapper(self)

    def __getstate__(self) -> Dict[str, Any]:
        return {'entity': self.entity, 'mapper': self.mapper, 'alias': self.selectable, 'name': self.name, 'adapt_on_names': self._adapt_on_names, 'with_polymorphic_mappers': self.with_polymorphic_mappers, 'with_polymorphic_discriminator': self.polymorphic_on, 'base_alias': self._base_alias(), 'use_mapper_path': self._use_mapper_path, 'represents_outer_join': self.represents_outer_join, 'nest_adapters': self._nest_adapters}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__init__(state['entity'], state['mapper'], state['alias'], state['name'], state['with_polymorphic_mappers'], state['with_polymorphic_discriminator'], state['base_alias'], state['use_mapper_path'], state['adapt_on_names'], state['represents_outer_join'], state['nest_adapters'])

    def _merge_with(self, other: AliasedInsp[_O]) -> AliasedInsp[_O]:
        primary_mapper = other.mapper
        assert self.mapper is primary_mapper
        our_classes = util.to_set((mp.class_ for mp in self.with_polymorphic_mappers))
        new_classes = {mp.class_ for mp in other.with_polymorphic_mappers}
        if our_classes == new_classes:
            return other
        else:
            classes = our_classes.union(new_classes)
        mappers, selectable = primary_mapper._with_polymorphic_args(classes, None, innerjoin=not other.represents_outer_join)
        selectable = selectable._anonymous_fromclause(flat=True)
        return AliasedClass(primary_mapper, selectable, with_polymorphic_mappers=mappers, with_polymorphic_discriminator=other.polymorphic_on, use_mapper_path=other._use_mapper_path, represents_outer_join=other.represents_outer_join)._aliased_insp

    def _adapt_element(self, expr: _ORMCOLEXPR, key: Optional[str]=None) -> _ORMCOLEXPR:
        assert isinstance(expr, ColumnElement)
        d: Dict[str, Any] = {'parententity': self, 'parentmapper': self.mapper}
        if key:
            d['proxy_key'] = key
        return self._adapter.traverse(expr)._annotate(d)._set_propagate_attrs({'compile_state_plugin': 'orm', 'plugin_subject': self})
    if TYPE_CHECKING:

        def _orm_adapt_element(self, obj: _CE, key: Optional[str]=None) -> _CE:
            ...
    else:
        _orm_adapt_element = _adapt_element

    def _entity_for_mapper(self, mapper):
        self_poly = self.with_polymorphic_mappers
        if mapper in self_poly:
            if mapper is self.mapper:
                return self
            else:
                return getattr(self.entity, mapper.class_.__name__)._aliased_insp
        elif mapper.isa(self.mapper):
            return self
        else:
            assert False, "mapper %s doesn't correspond to %s" % (mapper, self)

    def _memoized_attr__get_clause(self):
        onclause, replacemap = self.mapper._get_clause
        return (self._adapter.traverse(onclause), {self._adapter.traverse(col): param for col, param in replacemap.items()})

    def _memoized_attr__memoized_values(self):
        return {}

    def _memoized_attr__all_column_expressions(self):
        if self._is_with_polymorphic:
            cols_plus_keys = self.mapper._columns_plus_keys([ent.mapper for ent in self._with_polymorphic_entities])
        else:
            cols_plus_keys = self.mapper._columns_plus_keys()
        cols_plus_keys = [(key, self._adapt_element(col)) for key, col in cols_plus_keys]
        return ColumnCollection(cols_plus_keys)

    def _memo(self, key, callable_, *args, **kw):
        if key in self._memoized_values:
            return self._memoized_values[key]
        else:
            self._memoized_values[key] = value = callable_(*args, **kw)
            return value

    def __repr__(self):
        if self.with_polymorphic_mappers:
            with_poly = '(%s)' % ', '.join((mp.class_.__name__ for mp in self.with_polymorphic_mappers))
        else:
            with_poly = ''
        return '<AliasedInsp at 0x%x; %s%s>' % (id(self), self.class_.__name__, with_poly)

    def __str__(self):
        if self._is_with_polymorphic:
            return 'with_polymorphic(%s, [%s])' % (self._target.__name__, ', '.join((mp.class_.__name__ for mp in self.with_polymorphic_mappers if mp is not self.mapper)))
        else:
            return 'aliased(%s)' % (self._target.__name__,)