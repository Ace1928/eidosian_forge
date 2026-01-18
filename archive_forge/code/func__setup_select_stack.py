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
def _setup_select_stack(self, select, compile_state, entry, asfrom, lateral, compound_index):
    correlate_froms = entry['correlate_froms']
    asfrom_froms = entry['asfrom_froms']
    if compound_index == 0:
        entry['select_0'] = select
    elif compound_index:
        select_0 = entry['select_0']
        numcols = len(select_0._all_selected_columns)
        if len(compile_state.columns_plus_names) != numcols:
            raise exc.CompileError('All selectables passed to CompoundSelect must have identical numbers of columns; select #%d has %d columns, select #%d has %d' % (1, numcols, compound_index + 1, len(select._all_selected_columns)))
    if asfrom and (not lateral):
        froms = compile_state._get_display_froms(explicit_correlate_froms=correlate_froms.difference(asfrom_froms), implicit_correlate_froms=())
    else:
        froms = compile_state._get_display_froms(explicit_correlate_froms=correlate_froms, implicit_correlate_froms=asfrom_froms)
    new_correlate_froms = set(_from_objects(*froms))
    all_correlate_froms = new_correlate_froms.union(correlate_froms)
    new_entry: _CompilerStackEntry = {'asfrom_froms': new_correlate_froms, 'correlate_froms': all_correlate_froms, 'selectable': select, 'compile_state': compile_state}
    self.stack.append(new_entry)
    return froms