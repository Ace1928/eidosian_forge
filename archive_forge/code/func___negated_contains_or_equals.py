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
def __negated_contains_or_equals(self, other: Any) -> ColumnElement[bool]:
    if self.prop.direction == MANYTOONE:
        state = attributes.instance_state(other)

        def state_bindparam(local_col: ColumnElement[Any], state: InstanceState[Any], remote_col: ColumnElement[Any]) -> BindParameter[Any]:
            dict_ = state.dict
            return sql.bindparam(local_col.key, type_=local_col.type, unique=True, callable_=self.prop._get_attr_w_warn_on_none(self.prop.mapper, state, dict_, remote_col))

        def adapt(col: _CE) -> _CE:
            if self.adapter:
                return self.adapter(col)
            else:
                return col
        if self.property._use_get:
            return sql.and_(*[sql.or_(adapt(x) != state_bindparam(adapt(x), state, y), adapt(x) == None) for x, y in self.property.local_remote_pairs])
    criterion = sql.and_(*[x == y for x, y in zip(self.property.mapper.primary_key, self.property.mapper.primary_key_from_instance(other))])
    return ~self._criterion_exists(criterion)