from __future__ import annotations
import collections
import collections.abc as collections_abc
import contextlib
from enum import IntEnum
import functools
import itertools
import operator
import re
from time import perf_counter
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import FrozenSet
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import Pattern
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from . import base
from . import coercions
from . import crud
from . import elements
from . import functions
from . import operators
from . import roles
from . import schema
from . import selectable
from . import sqltypes
from . import util as sql_util
from ._typing import is_column_element
from ._typing import is_dml
from .base import _de_clone
from .base import _from_objects
from .base import _NONE_NAME
from .base import _SentinelDefaultCharacterization
from .base import Executable
from .base import NO_ARG
from .elements import ClauseElement
from .elements import quoted_name
from .schema import Column
from .sqltypes import TupleType
from .type_api import TypeEngine
from .visitors import prefix_anon_map
from .visitors import Visitable
from .. import exc
from .. import util
from ..util import FastIntFlag
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import TypedDict
class _InsertManyValues(NamedTuple):
    """represents state to use for executing an "insertmanyvalues" statement.

    The primary consumers of this object are the
    :meth:`.SQLCompiler._deliver_insertmanyvalues_batches` and
    :meth:`.DefaultDialect._deliver_insertmanyvalues_batches` methods.

    .. versionadded:: 2.0

    """
    is_default_expr: bool
    'if True, the statement is of the form\n    ``INSERT INTO TABLE DEFAULT VALUES``, and can\'t be rewritten as a "batch"\n\n    '
    single_values_expr: str
    'The rendered "values" clause of the INSERT statement.\n\n    This is typically the parenthesized section e.g. "(?, ?, ?)" or similar.\n    The insertmanyvalues logic uses this string as a search and replace\n    target.\n\n    '
    insert_crud_params: List[crud._CrudParamElementStr]
    'List of Column / bind names etc. used while rewriting the statement'
    num_positional_params_counted: int
    "the number of bound parameters in a single-row statement.\n\n    This count may be larger or smaller than the actual number of columns\n    targeted in the INSERT, as it accommodates for SQL expressions\n    in the values list that may have zero or more parameters embedded\n    within them.\n\n    This count is part of what's used to organize rewritten parameter lists\n    when batching.\n\n    "
    sort_by_parameter_order: bool = False
    'if the deterministic_returnined_order parameter were used on the\n    insert.\n\n    All of the attributes following this will only be used if this is True.\n\n    '
    includes_upsert_behaviors: bool = False
    'if True, we have to accommodate for upsert behaviors.\n\n    This will in some cases downgrade "insertmanyvalues" that requests\n    deterministic ordering.\n\n    '
    sentinel_columns: Optional[Sequence[Column[Any]]] = None
    'List of sentinel columns that were located.\n\n    This list is only here if the INSERT asked for\n    sort_by_parameter_order=True,\n    and dialect-appropriate sentinel columns were located.\n\n    .. versionadded:: 2.0.10\n\n    '
    num_sentinel_columns: int = 0
    'how many sentinel columns are in the above list, if any.\n\n    This is the same as\n    ``len(sentinel_columns) if sentinel_columns is not None else 0``\n\n    '
    sentinel_param_keys: Optional[Sequence[str]] = None
    'parameter str keys in each param dictionary / tuple\n    that would link to the client side "sentinel" values for that row, which\n    we can use to match up parameter sets to result rows.\n\n    This is only present if sentinel_columns is present and the INSERT\n    statement actually refers to client side values for these sentinel\n    columns.\n\n    .. versionadded:: 2.0.10\n\n    .. versionchanged:: 2.0.29 - the sequence is now string dictionary keys\n       only, used against the "compiled parameteters" collection before\n       the parameters were converted by bound parameter processors\n\n    '
    implicit_sentinel: bool = False
    'if True, we have exactly one sentinel column and it uses a server side\n    value, currently has to generate an incrementing integer value.\n\n    The dialect in question would have asserted that it supports receiving\n    these values back and sorting on that value as a means of guaranteeing\n    correlation with the incoming parameter list.\n\n    .. versionadded:: 2.0.10\n\n    '
    embed_values_counter: bool = False
    'Whether to embed an incrementing integer counter in each parameter\n    set within the VALUES clause as parameters are batched over.\n\n    This is only used for a specific INSERT..SELECT..VALUES..RETURNING syntax\n    where a subquery is used to produce value tuples.  Current support\n    includes PostgreSQL, Microsoft SQL Server.\n\n    .. versionadded:: 2.0.10\n\n    '