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
def _determine_direction(self) -> None:
    """Determine if this relationship is one to many, many to one,
        many to many.

        """
    if self.secondaryjoin is not None:
        self.direction = MANYTOMANY
    else:
        parentcols = util.column_set(self.parent_persist_selectable.c)
        targetcols = util.column_set(self.child_persist_selectable.c)
        onetomany_fk = targetcols.intersection(self.foreign_key_columns)
        manytoone_fk = parentcols.intersection(self.foreign_key_columns)
        if onetomany_fk and manytoone_fk:
            onetomany_local = self._gather_columns_with_annotation(self.primaryjoin, 'remote', 'foreign')
            manytoone_local = {c for c in self._gather_columns_with_annotation(self.primaryjoin, 'foreign') if 'remote' not in c._annotations}
            if onetomany_local and manytoone_local:
                self_equated = self.remote_columns.intersection(self.local_columns)
                onetomany_local = onetomany_local.difference(self_equated)
                manytoone_local = manytoone_local.difference(self_equated)
            if onetomany_local and (not manytoone_local):
                self.direction = ONETOMANY
            elif manytoone_local and (not onetomany_local):
                self.direction = MANYTOONE
            else:
                raise sa_exc.ArgumentError("Can't determine relationship direction for relationship '%s' - foreign key columns within the join condition are present in both the parent and the child's mapped tables.  Ensure that only those columns referring to a parent column are marked as foreign, either via the foreign() annotation or via the foreign_keys argument." % self.prop)
        elif onetomany_fk:
            self.direction = ONETOMANY
        elif manytoone_fk:
            self.direction = MANYTOONE
        else:
            raise sa_exc.ArgumentError("Can't determine relationship direction for relationship '%s' - foreign key columns are present in neither the parent nor the child's mapped tables" % self.prop)