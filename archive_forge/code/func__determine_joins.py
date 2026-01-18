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
def _determine_joins(self) -> None:
    """Determine the 'primaryjoin' and 'secondaryjoin' attributes,
        if not passed to the constructor already.

        This is based on analysis of the foreign key relationships
        between the parent and target mapped selectables.

        """
    if self.secondaryjoin is not None and self.secondary is None:
        raise sa_exc.ArgumentError('Property %s specified with secondary join condition but no secondary argument' % self.prop)
    try:
        consider_as_foreign_keys = self.consider_as_foreign_keys or None
        if self.secondary is not None:
            if self.secondaryjoin is None:
                self.secondaryjoin = join_condition(self.child_persist_selectable, self.secondary, a_subset=self.child_local_selectable, consider_as_foreign_keys=consider_as_foreign_keys)
            if self.primaryjoin_initial is None:
                self.primaryjoin = join_condition(self.parent_persist_selectable, self.secondary, a_subset=self.parent_local_selectable, consider_as_foreign_keys=consider_as_foreign_keys)
            else:
                self.primaryjoin = self.primaryjoin_initial
        elif self.primaryjoin_initial is None:
            self.primaryjoin = join_condition(self.parent_persist_selectable, self.child_persist_selectable, a_subset=self.parent_local_selectable, consider_as_foreign_keys=consider_as_foreign_keys)
        else:
            self.primaryjoin = self.primaryjoin_initial
    except sa_exc.NoForeignKeysError as nfe:
        if self.secondary is not None:
            raise sa_exc.NoForeignKeysError("Could not determine join condition between parent/child tables on relationship %s - there are no foreign keys linking these tables via secondary table '%s'.  Ensure that referencing columns are associated with a ForeignKey or ForeignKeyConstraint, or specify 'primaryjoin' and 'secondaryjoin' expressions." % (self.prop, self.secondary)) from nfe
        else:
            raise sa_exc.NoForeignKeysError("Could not determine join condition between parent/child tables on relationship %s - there are no foreign keys linking these tables.  Ensure that referencing columns are associated with a ForeignKey or ForeignKeyConstraint, or specify a 'primaryjoin' expression." % self.prop) from nfe
    except sa_exc.AmbiguousForeignKeysError as afe:
        if self.secondary is not None:
            raise sa_exc.AmbiguousForeignKeysError("Could not determine join condition between parent/child tables on relationship %s - there are multiple foreign key paths linking the tables via secondary table '%s'.  Specify the 'foreign_keys' argument, providing a list of those columns which should be counted as containing a foreign key reference from the secondary table to each of the parent and child tables." % (self.prop, self.secondary)) from afe
        else:
            raise sa_exc.AmbiguousForeignKeysError("Could not determine join condition between parent/child tables on relationship %s - there are multiple foreign key paths linking the tables.  Specify the 'foreign_keys' argument, providing a list of those columns which should be counted as containing a foreign key reference to the parent table." % self.prop) from afe