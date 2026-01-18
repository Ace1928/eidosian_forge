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
def _get_attr_w_warn_on_none(self, mapper: Mapper[Any], state: InstanceState[Any], dict_: _InstanceDict, column: ColumnElement[Any]) -> Callable[[], Any]:
    """Create the callable that is used in a many-to-one expression.

        E.g.::

            u1 = s.query(User).get(5)

            expr = Address.user == u1

        Above, the SQL should be "address.user_id = 5". The callable
        returned by this method produces the value "5" based on the identity
        of ``u1``.

        """
    prop = mapper.get_property_by_column(column)
    state._track_last_known_value(prop.key)
    lkv_fixed = state._last_known_values

    def _go() -> Any:
        assert lkv_fixed is not None
        last_known = to_return = lkv_fixed[prop.key]
        existing_is_available = last_known is not LoaderCallableStatus.NO_VALUE
        current_value = mapper._get_state_attr_by_column(state, dict_, column, passive=PassiveFlag.PASSIVE_OFF if state.persistent else PassiveFlag.PASSIVE_NO_FETCH ^ PassiveFlag.INIT_OK)
        if current_value is LoaderCallableStatus.NEVER_SET:
            if not existing_is_available:
                raise sa_exc.InvalidRequestError("Can't resolve value for column %s on object %s; no value has been set for this column" % (column, state_str(state)))
        elif current_value is LoaderCallableStatus.PASSIVE_NO_RESULT:
            if not existing_is_available:
                raise sa_exc.InvalidRequestError("Can't resolve value for column %s on object %s; the object is detached and the value was expired" % (column, state_str(state)))
        else:
            to_return = current_value
        if to_return is None:
            util.warn('Got None for value of column %s; this is unsupported for a relationship comparison and will not currently produce an IS comparison (but may in a future release)' % column)
        return to_return
    return _go