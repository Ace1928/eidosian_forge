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
def _process_numeric(self):
    assert self._numeric_binds
    assert self.state is CompilerState.STRING_APPLIED
    num = 1
    param_pos: Dict[str, str] = {}
    order: Iterable[str]
    if self._insertmanyvalues and self._values_bindparam is not None:
        order = itertools.chain((name for name in self.bind_names.values() if name not in self._values_bindparam), self.bind_names.values())
    else:
        order = self.bind_names.values()
    for bind_name in order:
        if bind_name in param_pos:
            continue
        bind = self.binds[bind_name]
        if bind in self.post_compile_params or bind in self.literal_execute_params:
            param_pos[bind_name] = None
        else:
            ph = f'{self._numeric_binds_identifier_char}{num}'
            num += 1
            param_pos[bind_name] = ph
    self.next_numeric_pos = num
    self.positiontup = list(param_pos)
    if self.escaped_bind_names:
        len_before = len(param_pos)
        param_pos = {self.escaped_bind_names.get(name, name): pos for name, pos in param_pos.items()}
        assert len(param_pos) == len_before
    self.string = self._pyformat_pattern.sub(lambda m: param_pos[m.group(1)], self.string)
    if self._insertmanyvalues:
        single_values_expr = self._insertmanyvalues.single_values_expr % param_pos
        insert_crud_params = [(v[0], v[1], '%s', v[3]) for v in self._insertmanyvalues.insert_crud_params]
        self._insertmanyvalues = self._insertmanyvalues._replace(single_values_expr=single_values_expr, insert_crud_params=insert_crud_params)