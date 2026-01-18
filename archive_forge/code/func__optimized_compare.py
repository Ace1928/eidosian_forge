from __future__ import annotations
import collections
from collections import abc
import dataclasses
import inspect as _py_inspect
import itertools
import re
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
import weakref
from . import attributes
from . import strategy_options
from ._typing import insp_is_aliased_class
from ._typing import is_has_collection_adapter
from .base import _DeclarativeMapped
from .base import _is_mapped_class
from .base import class_mapper
from .base import DynamicMapped
from .base import LoaderCallableStatus
from .base import PassiveFlag
from .base import state_str
from .base import WriteOnlyMapped
from .interfaces import _AttributeOptions
from .interfaces import _IntrospectsAnnotations
from .interfaces import MANYTOMANY
from .interfaces import MANYTOONE
from .interfaces import ONETOMANY
from .interfaces import PropComparator
from .interfaces import RelationshipDirection
from .interfaces import StrategizedProperty
from .util import _orm_annotate
from .util import _orm_deannotate
from .util import CascadeOptions
from .. import exc as sa_exc
from .. import Exists
from .. import log
from .. import schema
from .. import sql
from .. import util
from ..inspection import inspect
from ..sql import coercions
from ..sql import expression
from ..sql import operators
from ..sql import roles
from ..sql import visitors
from ..sql._typing import _ColumnExpressionArgument
from ..sql._typing import _HasClauseElement
from ..sql.annotation import _safe_annotate
from ..sql.elements import ColumnClause
from ..sql.elements import ColumnElement
from ..sql.util import _deep_annotate
from ..sql.util import _deep_deannotate
from ..sql.util import _shallow_annotate
from ..sql.util import adapt_criterion_to_null
from ..sql.util import ClauseAdapter
from ..sql.util import join_condition
from ..sql.util import selectables_overlap
from ..sql.util import visit_binary_product
from ..util.typing import de_optionalize_union_types
from ..util.typing import Literal
from ..util.typing import resolve_name_to_real_class_name
def _optimized_compare(self, state: Any, value_is_parent: bool=False, adapt_source: Optional[_CoreAdapterProto]=None, alias_secondary: bool=True) -> ColumnElement[bool]:
    if state is not None:
        try:
            state = inspect(state)
        except sa_exc.NoInspectionAvailable:
            state = None
        if state is None or not getattr(state, 'is_instance', False):
            raise sa_exc.ArgumentError('Mapped instance expected for relationship comparison to object.   Classes, queries and other SQL elements are not accepted in this context; for comparison with a subquery, use %s.has(**criteria).' % self)
    reverse_direction = not value_is_parent
    if state is None:
        return self._lazy_none_clause(reverse_direction, adapt_source=adapt_source)
    if not reverse_direction:
        criterion, bind_to_col = (self._lazy_strategy._lazywhere, self._lazy_strategy._bind_to_col)
    else:
        criterion, bind_to_col = (self._lazy_strategy._rev_lazywhere, self._lazy_strategy._rev_bind_to_col)
    if reverse_direction:
        mapper = self.mapper
    else:
        mapper = self.parent
    dict_ = attributes.instance_dict(state.obj())

    def visit_bindparam(bindparam: BindParameter[Any]) -> None:
        if bindparam._identifying_key in bind_to_col:
            bindparam.callable = self._get_attr_w_warn_on_none(mapper, state, dict_, bind_to_col[bindparam._identifying_key])
    if self.secondary is not None and alias_secondary:
        criterion = ClauseAdapter(self.secondary._anonymous_fromclause()).traverse(criterion)
    criterion = visitors.cloned_traverse(criterion, {}, {'bindparam': visit_bindparam})
    if adapt_source:
        criterion = adapt_source(criterion)
    return criterion