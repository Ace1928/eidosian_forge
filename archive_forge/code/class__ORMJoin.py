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
class _ORMJoin(expression.Join):
    """Extend Join to support ORM constructs as input."""
    __visit_name__ = expression.Join.__visit_name__
    inherit_cache = True

    def __init__(self, left: _FromClauseArgument, right: _FromClauseArgument, onclause: Optional[_OnClauseArgument]=None, isouter: bool=False, full: bool=False, _left_memo: Optional[Any]=None, _right_memo: Optional[Any]=None, _extra_criteria: Tuple[ColumnElement[bool], ...]=()):
        left_info = cast('Union[FromClause, _InternalEntityType[Any]]', inspection.inspect(left))
        right_info = cast('Union[FromClause, _InternalEntityType[Any]]', inspection.inspect(right))
        adapt_to = right_info.selectable
        self._left_memo = _left_memo
        self._right_memo = _right_memo
        if isinstance(onclause, attributes.QueryableAttribute):
            if TYPE_CHECKING:
                assert isinstance(onclause.comparator, RelationshipProperty.Comparator)
            on_selectable = onclause.comparator._source_selectable()
            prop = onclause.property
            _extra_criteria += onclause._extra_criteria
        elif isinstance(onclause, MapperProperty):
            prop = onclause
            on_selectable = prop.parent.selectable
        else:
            prop = None
            on_selectable = None
        left_selectable = left_info.selectable
        if prop:
            adapt_from: Optional[FromClause]
            if sql_util.clause_is_present(on_selectable, left_selectable):
                adapt_from = on_selectable
            else:
                assert isinstance(left_selectable, FromClause)
                adapt_from = left_selectable
            pj, sj, source, dest, secondary, target_adapter = prop._create_joins(source_selectable=adapt_from, dest_selectable=adapt_to, source_polymorphic=True, of_type_entity=right_info, alias_secondary=True, extra_criteria=_extra_criteria)
            if sj is not None:
                if isouter:
                    right = sql.join(secondary, right, sj)
                    onclause = pj
                else:
                    left = sql.join(left, secondary, pj, isouter)
                    onclause = sj
            else:
                onclause = pj
            self._target_adapter = target_adapter
        if is_selectable(left_info):
            parententity = left_selectable._annotations.get('parententity', None)
        elif insp_is_mapper(left_info) or insp_is_aliased_class(left_info):
            parententity = left_info
        else:
            parententity = None
        if parententity is not None:
            self._annotations = self._annotations.union({'parententity': parententity})
        augment_onclause = bool(_extra_criteria) and (not prop)
        expression.Join.__init__(self, left, right, onclause, isouter, full)
        assert self.onclause is not None
        if augment_onclause:
            self.onclause &= sql.and_(*_extra_criteria)
        if not prop and getattr(right_info, 'mapper', None) and right_info.mapper.single:
            right_info = cast('_InternalEntityType[Any]', right_info)
            single_crit = right_info.mapper._single_table_criterion
            if single_crit is not None:
                if insp_is_aliased_class(right_info):
                    single_crit = right_info._adapter.traverse(single_crit)
                self.onclause = self.onclause & single_crit

    def _splice_into_center(self, other):
        """Splice a join into the center.

        Given join(a, b) and join(b, c), return join(a, b).join(c)

        """
        leftmost = other
        while isinstance(leftmost, sql.Join):
            leftmost = leftmost.left
        assert self.right is leftmost
        left = _ORMJoin(self.left, other.left, self.onclause, isouter=self.isouter, _left_memo=self._left_memo, _right_memo=other._left_memo)
        return _ORMJoin(left, other.right, other.onclause, isouter=other.isouter, _right_memo=other._right_memo)

    def join(self, right: _FromClauseArgument, onclause: Optional[_OnClauseArgument]=None, isouter: bool=False, full: bool=False) -> _ORMJoin:
        return _ORMJoin(self, right, onclause, full=full, isouter=isouter)

    def outerjoin(self, right: _FromClauseArgument, onclause: Optional[_OnClauseArgument]=None, full: bool=False) -> _ORMJoin:
        return _ORMJoin(self, right, onclause, isouter=True, full=full)