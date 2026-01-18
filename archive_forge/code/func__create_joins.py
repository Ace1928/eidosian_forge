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
def _create_joins(self, source_polymorphic: bool=False, source_selectable: Optional[FromClause]=None, dest_selectable: Optional[FromClause]=None, of_type_entity: Optional[_InternalEntityType[Any]]=None, alias_secondary: bool=False, extra_criteria: Tuple[ColumnElement[bool], ...]=()) -> Tuple[ColumnElement[bool], Optional[ColumnElement[bool]], FromClause, FromClause, Optional[FromClause], Optional[ClauseAdapter]]:
    aliased = False
    if alias_secondary and self.secondary is not None:
        aliased = True
    if source_selectable is None:
        if source_polymorphic and self.parent.with_polymorphic:
            source_selectable = self.parent._with_polymorphic_selectable
    if of_type_entity:
        dest_mapper = of_type_entity.mapper
        if dest_selectable is None:
            dest_selectable = of_type_entity.selectable
            aliased = True
    else:
        dest_mapper = self.mapper
    if dest_selectable is None:
        dest_selectable = self.entity.selectable
        if self.mapper.with_polymorphic:
            aliased = True
        if self._is_self_referential and source_selectable is None:
            dest_selectable = dest_selectable._anonymous_fromclause()
            aliased = True
    elif dest_selectable is not self.mapper._with_polymorphic_selectable or self.mapper.with_polymorphic:
        aliased = True
    single_crit = dest_mapper._single_table_criterion
    aliased = aliased or (source_selectable is not None and (source_selectable is not self.parent._with_polymorphic_selectable or source_selectable._is_subquery))
    primaryjoin, secondaryjoin, secondary, target_adapter, dest_selectable = self._join_condition.join_targets(source_selectable, dest_selectable, aliased, single_crit, extra_criteria)
    if source_selectable is None:
        source_selectable = self.parent.local_table
    if dest_selectable is None:
        dest_selectable = self.entity.local_table
    return (primaryjoin, secondaryjoin, source_selectable, dest_selectable, secondary, target_adapter)