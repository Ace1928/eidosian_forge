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
def _literal_execute_expanding_parameter_literal_binds(self, parameter, values, bind_expression_template=None):
    typ_dialect_impl = parameter.type._unwrapped_dialect_impl(self.dialect)
    if not values:
        if typ_dialect_impl._is_tuple_type:
            replacement_expression = ('VALUES ' if self.dialect.tuple_in_values else '') + self.visit_empty_set_op_expr(parameter.type.types, parameter.expand_op)
        else:
            replacement_expression = self.visit_empty_set_op_expr([parameter.type], parameter.expand_op)
    elif typ_dialect_impl._is_tuple_type or (typ_dialect_impl._isnull and isinstance(values[0], collections_abc.Sequence) and (not isinstance(values[0], (str, bytes)))):
        if typ_dialect_impl._has_bind_expression:
            raise NotImplementedError('bind_expression() on TupleType not supported with literal_binds')
        replacement_expression = ('VALUES ' if self.dialect.tuple_in_values else '') + ', '.join(('(%s)' % ', '.join((self.render_literal_value(value, param_type) for value, param_type in zip(tuple_element, parameter.type.types))) for i, tuple_element in enumerate(values)))
    elif bind_expression_template:
        post_compile_pattern = self._post_compile_pattern
        m = post_compile_pattern.search(bind_expression_template)
        assert m and m.group(2), 'unexpected format for expanding parameter'
        tok = m.group(2).split('~~')
        be_left, be_right = (tok[1], tok[3])
        replacement_expression = ', '.join(('%s%s%s' % (be_left, self.render_literal_value(value, parameter.type), be_right) for value in values))
    else:
        replacement_expression = ', '.join((self.render_literal_value(value, parameter.type) for value in values))
    return ((), replacement_expression)