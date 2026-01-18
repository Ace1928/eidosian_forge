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
class _InsertManyValuesBatch(NamedTuple):
    """represents an individual batch SQL statement for insertmanyvalues.

    This is passed through the
    :meth:`.SQLCompiler._deliver_insertmanyvalues_batches` and
    :meth:`.DefaultDialect._deliver_insertmanyvalues_batches` methods out
    to the :class:`.Connection` within the
    :meth:`.Connection._exec_insertmany_context` method.

    .. versionadded:: 2.0.10

    """
    replaced_statement: str
    replaced_parameters: _DBAPIAnyExecuteParams
    processed_setinputsizes: Optional[_GenericSetInputSizesType]
    batch: Sequence[_DBAPISingleExecuteParams]
    sentinel_values: Sequence[Tuple[Any, ...]]
    current_batch_size: int
    batchnum: int
    total_batches: int
    rows_sorted: bool
    is_downgraded: bool