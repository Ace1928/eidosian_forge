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
def _render_schema_translates(self, statement, schema_translate_map):
    d = schema_translate_map
    if None in d:
        if not self._includes_none_schema_translate:
            raise exc.InvalidRequestError('schema translate map which previously did not have `None` present as a key now has `None` present; compiled statement may lack adequate placeholders.  Please use consistent keys in successive schema_translate_map dictionaries.')
        d['_none'] = d[None]

    def replace(m):
        name = m.group(2)
        if name in d:
            effective_schema = d[name]
        else:
            if name in (None, '_none'):
                raise exc.InvalidRequestError("schema translate map which previously had `None` present as a key now no longer has it present; don't know how to apply schema for compiled statement. Please use consistent keys in successive schema_translate_map dictionaries.")
            effective_schema = name
        if not effective_schema:
            effective_schema = self.dialect.default_schema_name
            if not effective_schema:
                raise exc.CompileError("Dialect has no default schema name; can't use None as dynamic schema target.")
        return self.quote_schema(effective_schema)
    return re.sub('(__\\[SCHEMA_([^\\]]+)\\])', replace, statement)