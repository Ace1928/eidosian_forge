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
def _check_cascade_settings(self, cascade: CascadeOptions) -> None:
    if cascade.delete_orphan and (not self.single_parent) and (self.direction is MANYTOMANY or self.direction is MANYTOONE):
        raise sa_exc.ArgumentError('For %(direction)s relationship %(rel)s, delete-orphan cascade is normally configured only on the "one" side of a one-to-many relationship, and not on the "many" side of a many-to-one or many-to-many relationship.  To force this relationship to allow a particular "%(relatedcls)s" object to be referenced by only a single "%(clsname)s" object at a time via the %(rel)s relationship, which would allow delete-orphan cascade to take place in this direction, set the single_parent=True flag.' % {'rel': self, 'direction': 'many-to-one' if self.direction is MANYTOONE else 'many-to-many', 'clsname': self.parent.class_.__name__, 'relatedcls': self.mapper.class_.__name__}, code='bbf0')
    if self.passive_deletes == 'all' and ('delete' in cascade or 'delete-orphan' in cascade):
        raise sa_exc.ArgumentError("On %s, can't set passive_deletes='all' in conjunction with 'delete' or 'delete-orphan' cascade" % self)
    if cascade.delete_orphan:
        self.mapper.primary_mapper()._delete_orphans.append((self.key, self.parent.class_))