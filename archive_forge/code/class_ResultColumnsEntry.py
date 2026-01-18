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
class ResultColumnsEntry(NamedTuple):
    """Tracks a column expression that is expected to be represented
    in the result rows for this statement.

    This normally refers to the columns clause of a SELECT statement
    but may also refer to a RETURNING clause, as well as for dialect-specific
    emulations.

    """
    keyname: str
    "string name that's expected in cursor.description"
    name: str
    'column name, may be labeled'
    objects: Tuple[Any, ...]
    'sequence of objects that should be able to locate this column\n    in a RowMapping.  This is typically string names and aliases\n    as well as Column objects.\n\n    '
    type: TypeEngine[Any]
    'Datatype to be associated with this column.   This is where\n    the "result processing" logic directly links the compiled statement\n    to the rows that come back from the cursor.\n\n    '