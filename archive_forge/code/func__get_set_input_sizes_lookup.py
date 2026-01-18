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
@util.memoized_instancemethod
def _get_set_input_sizes_lookup(self):
    dialect = self.dialect
    include_types = dialect.include_set_input_sizes
    exclude_types = dialect.exclude_set_input_sizes
    dbapi = dialect.dbapi

    def lookup_type(typ):
        dbtype = typ._unwrapped_dialect_impl(dialect).get_dbapi_type(dbapi)
        if dbtype is not None and (exclude_types is None or dbtype not in exclude_types) and (include_types is None or dbtype in include_types):
            return dbtype
        else:
            return None
    inputsizes = {}
    literal_execute_params = self.literal_execute_params
    for bindparam in self.bind_names:
        if bindparam in literal_execute_params:
            continue
        if bindparam.type._is_tuple_type:
            inputsizes[bindparam] = [lookup_type(typ) for typ in cast(TupleType, bindparam.type).types]
        else:
            inputsizes[bindparam] = lookup_type(bindparam.type)
    return inputsizes