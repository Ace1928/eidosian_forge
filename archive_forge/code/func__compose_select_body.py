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
def _compose_select_body(self, text, select, compile_state, inner_columns, froms, byfrom, toplevel, kwargs):
    text += ', '.join(inner_columns)
    if self.linting & COLLECT_CARTESIAN_PRODUCTS:
        from_linter = FromLinter({}, set())
        warn_linting = self.linting & WARN_LINTING
        if toplevel:
            self.from_linter = from_linter
    else:
        from_linter = None
        warn_linting = False
    if not inner_columns:
        text = text.rstrip()
    if froms:
        text += ' \nFROM '
        if select._hints:
            text += ', '.join([f._compiler_dispatch(self, asfrom=True, fromhints=byfrom, from_linter=from_linter, **kwargs) for f in froms])
        else:
            text += ', '.join([f._compiler_dispatch(self, asfrom=True, from_linter=from_linter, **kwargs) for f in froms])
    else:
        text += self.default_from()
    if select._where_criteria:
        t = self._generate_delimited_and_list(select._where_criteria, from_linter=from_linter, **kwargs)
        if t:
            text += ' \nWHERE ' + t
    if warn_linting:
        assert from_linter is not None
        from_linter.warn()
    if select._group_by_clauses:
        text += self.group_by_clause(select, **kwargs)
    if select._having_criteria:
        t = self._generate_delimited_and_list(select._having_criteria, **kwargs)
        if t:
            text += ' \nHAVING ' + t
    if select._order_by_clauses:
        text += self.order_by_clause(select, **kwargs)
    if select._has_row_limiting_clause:
        text += self._row_limit_clause(select, **kwargs)
    if select._for_update_arg is not None:
        text += self.for_update_clause(select, **kwargs)
    return text