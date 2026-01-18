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
def _generate_backref(self) -> None:
    """Interpret the 'backref' instruction to create a
        :func:`_orm.relationship` complementary to this one."""
    if self.parent.non_primary:
        return
    if self.backref is not None and (not self.back_populates):
        kwargs: Dict[str, Any]
        if isinstance(self.backref, str):
            backref_key, kwargs = (self.backref, {})
        else:
            backref_key, kwargs = self.backref
        mapper = self.mapper.primary_mapper()
        if not mapper.concrete:
            check = set(mapper.iterate_to_root()).union(mapper.self_and_descendants)
            for m in check:
                if m.has_property(backref_key) and (not m.concrete):
                    raise sa_exc.ArgumentError("Error creating backref '%s' on relationship '%s': property of that name exists on mapper '%s'" % (backref_key, self, m))
        if self.secondary is not None:
            pj = kwargs.pop('primaryjoin', self._join_condition.secondaryjoin_minus_local)
            sj = kwargs.pop('secondaryjoin', self._join_condition.primaryjoin_minus_local)
        else:
            pj = kwargs.pop('primaryjoin', self._join_condition.primaryjoin_reverse_remote)
            sj = kwargs.pop('secondaryjoin', None)
            if sj:
                raise sa_exc.InvalidRequestError("Can't assign 'secondaryjoin' on a backref against a non-secondary relationship.")
        foreign_keys = kwargs.pop('foreign_keys', self._user_defined_foreign_keys)
        parent = self.parent.primary_mapper()
        kwargs.setdefault('viewonly', self.viewonly)
        kwargs.setdefault('post_update', self.post_update)
        kwargs.setdefault('passive_updates', self.passive_updates)
        kwargs.setdefault('sync_backref', self.sync_backref)
        self.back_populates = backref_key
        relationship = RelationshipProperty(parent, self.secondary, primaryjoin=pj, secondaryjoin=sj, foreign_keys=foreign_keys, back_populates=self.key, **kwargs)
        mapper._configure_property(backref_key, relationship, warn_for_existing=True)
    if self.back_populates:
        self._add_reverse_property(self.back_populates)