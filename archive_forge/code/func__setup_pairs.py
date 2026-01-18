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
def _setup_pairs(self) -> None:
    sync_pairs: _MutableColumnPairs = []
    lrp: util.OrderedSet[Tuple[ColumnElement[Any], ColumnElement[Any]]] = util.OrderedSet([])
    secondary_sync_pairs: _MutableColumnPairs = []

    def go(joincond: ColumnElement[bool], collection: _MutableColumnPairs) -> None:

        def visit_binary(binary: BinaryExpression[Any], left: ColumnElement[Any], right: ColumnElement[Any]) -> None:
            if 'remote' in right._annotations and 'remote' not in left._annotations and self.can_be_synced_fn(left):
                lrp.add((left, right))
            elif 'remote' in left._annotations and 'remote' not in right._annotations and self.can_be_synced_fn(right):
                lrp.add((right, left))
            if binary.operator is operators.eq and self.can_be_synced_fn(left, right):
                if 'foreign' in right._annotations:
                    collection.append((left, right))
                elif 'foreign' in left._annotations:
                    collection.append((right, left))
        visit_binary_product(visit_binary, joincond)
    for joincond, collection in [(self.primaryjoin, sync_pairs), (self.secondaryjoin, secondary_sync_pairs)]:
        if joincond is None:
            continue
        go(joincond, collection)
    self.local_remote_pairs = self._deannotate_pairs(lrp)
    self.synchronize_pairs = self._deannotate_pairs(sync_pairs)
    self.secondary_synchronize_pairs = self._deannotate_pairs(secondary_sync_pairs)