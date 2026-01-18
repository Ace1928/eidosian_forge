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
class FromLinter(collections.namedtuple('FromLinter', ['froms', 'edges'])):
    """represents current state for the "cartesian product" detection
    feature."""

    def lint(self, start=None):
        froms = self.froms
        if not froms:
            return (None, None)
        edges = set(self.edges)
        the_rest = set(froms)
        if start is not None:
            start_with = start
            the_rest.remove(start_with)
        else:
            start_with = the_rest.pop()
        stack = collections.deque([start_with])
        while stack and the_rest:
            node = stack.popleft()
            the_rest.discard(node)
            to_remove = {edge for edge in edges if node in edge}
            stack.extendleft((edge[not edge.index(node)] for edge in to_remove))
            edges.difference_update(to_remove)
        if the_rest:
            return (the_rest, start_with)
        else:
            return (None, None)

    def warn(self, stmt_type='SELECT'):
        the_rest, start_with = self.lint()
        if the_rest:
            froms = the_rest
            if froms:
                template = '{stmt_type} statement has a cartesian product between FROM element(s) {froms} and FROM element "{start}".  Apply join condition(s) between each element to resolve.'
                froms_str = ', '.join((f'"{self.froms[from_]}"' for from_ in froms))
                message = template.format(stmt_type=stmt_type, froms=froms_str, start=self.froms[start_with])
                util.warn(message)