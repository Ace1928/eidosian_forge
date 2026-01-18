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
def _setup_join_conditions(self) -> None:
    self._join_condition = jc = JoinCondition(parent_persist_selectable=self.parent.persist_selectable, child_persist_selectable=self.entity.persist_selectable, parent_local_selectable=self.parent.local_table, child_local_selectable=self.entity.local_table, primaryjoin=self._init_args.primaryjoin.resolved, secondary=self._init_args.secondary.resolved, secondaryjoin=self._init_args.secondaryjoin.resolved, parent_equivalents=self.parent._equivalent_columns, child_equivalents=self.mapper._equivalent_columns, consider_as_foreign_keys=self._user_defined_foreign_keys, local_remote_pairs=self.local_remote_pairs, remote_side=self.remote_side, self_referential=self._is_self_referential, prop=self, support_sync=not self.viewonly, can_be_synced_fn=self._columns_are_mapped)
    self.primaryjoin = jc.primaryjoin
    self.secondaryjoin = jc.secondaryjoin
    self.secondary = jc.secondary
    self.direction = jc.direction
    self.local_remote_pairs = jc.local_remote_pairs
    self.remote_side = jc.remote_columns
    self.local_columns = jc.local_columns
    self.synchronize_pairs = jc.synchronize_pairs
    self._calculated_foreign_keys = jc.foreign_key_columns
    self.secondary_synchronize_pairs = jc.secondary_synchronize_pairs