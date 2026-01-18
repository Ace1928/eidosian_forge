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
class LoaderCriteriaOption(CriteriaOption):
    """Add additional WHERE criteria to the load for all occurrences of
    a particular entity.

    :class:`_orm.LoaderCriteriaOption` is invoked using the
    :func:`_orm.with_loader_criteria` function; see that function for
    details.

    .. versionadded:: 1.4

    """
    __slots__ = ('root_entity', 'entity', 'deferred_where_criteria', 'where_criteria', '_where_crit_orig', 'include_aliases', 'propagate_to_loaders')
    _traverse_internals = [('root_entity', visitors.ExtendedInternalTraversal.dp_plain_obj), ('entity', visitors.ExtendedInternalTraversal.dp_has_cache_key), ('where_criteria', visitors.InternalTraversal.dp_clauseelement), ('include_aliases', visitors.InternalTraversal.dp_boolean), ('propagate_to_loaders', visitors.InternalTraversal.dp_boolean)]
    root_entity: Optional[Type[Any]]
    entity: Optional[_InternalEntityType[Any]]
    where_criteria: Union[ColumnElement[bool], lambdas.DeferredLambdaElement]
    deferred_where_criteria: bool
    include_aliases: bool
    propagate_to_loaders: bool
    _where_crit_orig: Any

    def __init__(self, entity_or_base: _EntityType[Any], where_criteria: Union[_ColumnExpressionArgument[bool], Callable[[Any], _ColumnExpressionArgument[bool]]], loader_only: bool=False, include_aliases: bool=False, propagate_to_loaders: bool=True, track_closure_variables: bool=True):
        entity = cast('_InternalEntityType[Any]', inspection.inspect(entity_or_base, False))
        if entity is None:
            self.root_entity = cast('Type[Any]', entity_or_base)
            self.entity = None
        else:
            self.root_entity = None
            self.entity = entity
        self._where_crit_orig = where_criteria
        if callable(where_criteria):
            if self.root_entity is not None:
                wrap_entity = self.root_entity
            else:
                assert entity is not None
                wrap_entity = entity.entity
            self.deferred_where_criteria = True
            self.where_criteria = lambdas.DeferredLambdaElement(where_criteria, roles.WhereHavingRole, lambda_args=(_WrapUserEntity(wrap_entity),), opts=lambdas.LambdaOptions(track_closure_variables=track_closure_variables))
        else:
            self.deferred_where_criteria = False
            self.where_criteria = coercions.expect(roles.WhereHavingRole, where_criteria)
        self.include_aliases = include_aliases
        self.propagate_to_loaders = propagate_to_loaders

    @classmethod
    def _unreduce(cls, entity, where_criteria, include_aliases, propagate_to_loaders):
        return LoaderCriteriaOption(entity, where_criteria, include_aliases=include_aliases, propagate_to_loaders=propagate_to_loaders)

    def __reduce__(self):
        return (LoaderCriteriaOption._unreduce, (self.entity.class_ if self.entity else self.root_entity, self._where_crit_orig, self.include_aliases, self.propagate_to_loaders))

    def _all_mappers(self) -> Iterator[Mapper[Any]]:
        if self.entity:
            yield from self.entity.mapper.self_and_descendants
        else:
            assert self.root_entity
            stack = list(self.root_entity.__subclasses__())
            while stack:
                subclass = stack.pop(0)
                ent = cast('_InternalEntityType[Any]', inspection.inspect(subclass, raiseerr=False))
                if ent:
                    yield from ent.mapper.self_and_descendants
                else:
                    stack.extend(subclass.__subclasses__())

    def _should_include(self, compile_state: ORMCompileState) -> bool:
        if compile_state.select_statement._annotations.get('for_loader_criteria', None) is self:
            return False
        return True

    def _resolve_where_criteria(self, ext_info: _InternalEntityType[Any]) -> ColumnElement[bool]:
        if self.deferred_where_criteria:
            crit = cast('ColumnElement[bool]', self.where_criteria._resolve_with_args(ext_info.entity))
        else:
            crit = self.where_criteria
        assert isinstance(crit, ColumnElement)
        return sql_util._deep_annotate(crit, {'for_loader_criteria': self}, detect_subquery_cols=True, ind_cols_on_fromclause=True)

    def process_compile_state_replaced_entities(self, compile_state: ORMCompileState, mapper_entities: Iterable[_MapperEntity]) -> None:
        self.process_compile_state(compile_state)

    def process_compile_state(self, compile_state: ORMCompileState) -> None:
        """Apply a modification to a given :class:`.CompileState`."""
        self.get_global_criteria(compile_state.global_attributes)

    def get_global_criteria(self, attributes: Dict[Any, Any]) -> None:
        for mp in self._all_mappers():
            load_criteria = attributes.setdefault(('additional_entity_criteria', mp), [])
            load_criteria.append(self)