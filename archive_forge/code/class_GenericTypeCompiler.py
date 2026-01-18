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
class GenericTypeCompiler(TypeCompiler):

    def visit_FLOAT(self, type_, **kw):
        return 'FLOAT'

    def visit_DOUBLE(self, type_, **kw):
        return 'DOUBLE'

    def visit_DOUBLE_PRECISION(self, type_, **kw):
        return 'DOUBLE PRECISION'

    def visit_REAL(self, type_, **kw):
        return 'REAL'

    def visit_NUMERIC(self, type_, **kw):
        if type_.precision is None:
            return 'NUMERIC'
        elif type_.scale is None:
            return 'NUMERIC(%(precision)s)' % {'precision': type_.precision}
        else:
            return 'NUMERIC(%(precision)s, %(scale)s)' % {'precision': type_.precision, 'scale': type_.scale}

    def visit_DECIMAL(self, type_, **kw):
        if type_.precision is None:
            return 'DECIMAL'
        elif type_.scale is None:
            return 'DECIMAL(%(precision)s)' % {'precision': type_.precision}
        else:
            return 'DECIMAL(%(precision)s, %(scale)s)' % {'precision': type_.precision, 'scale': type_.scale}

    def visit_INTEGER(self, type_, **kw):
        return 'INTEGER'

    def visit_SMALLINT(self, type_, **kw):
        return 'SMALLINT'

    def visit_BIGINT(self, type_, **kw):
        return 'BIGINT'

    def visit_TIMESTAMP(self, type_, **kw):
        return 'TIMESTAMP'

    def visit_DATETIME(self, type_, **kw):
        return 'DATETIME'

    def visit_DATE(self, type_, **kw):
        return 'DATE'

    def visit_TIME(self, type_, **kw):
        return 'TIME'

    def visit_CLOB(self, type_, **kw):
        return 'CLOB'

    def visit_NCLOB(self, type_, **kw):
        return 'NCLOB'

    def _render_string_type(self, type_, name, length_override=None):
        text = name
        if length_override:
            text += '(%d)' % length_override
        elif type_.length:
            text += '(%d)' % type_.length
        if type_.collation:
            text += ' COLLATE "%s"' % type_.collation
        return text

    def visit_CHAR(self, type_, **kw):
        return self._render_string_type(type_, 'CHAR')

    def visit_NCHAR(self, type_, **kw):
        return self._render_string_type(type_, 'NCHAR')

    def visit_VARCHAR(self, type_, **kw):
        return self._render_string_type(type_, 'VARCHAR')

    def visit_NVARCHAR(self, type_, **kw):
        return self._render_string_type(type_, 'NVARCHAR')

    def visit_TEXT(self, type_, **kw):
        return self._render_string_type(type_, 'TEXT')

    def visit_UUID(self, type_, **kw):
        return 'UUID'

    def visit_BLOB(self, type_, **kw):
        return 'BLOB'

    def visit_BINARY(self, type_, **kw):
        return 'BINARY' + (type_.length and '(%d)' % type_.length or '')

    def visit_VARBINARY(self, type_, **kw):
        return 'VARBINARY' + (type_.length and '(%d)' % type_.length or '')

    def visit_BOOLEAN(self, type_, **kw):
        return 'BOOLEAN'

    def visit_uuid(self, type_, **kw):
        if not type_.native_uuid or not self.dialect.supports_native_uuid:
            return self._render_string_type(type_, 'CHAR', length_override=32)
        else:
            return self.visit_UUID(type_, **kw)

    def visit_large_binary(self, type_, **kw):
        return self.visit_BLOB(type_, **kw)

    def visit_boolean(self, type_, **kw):
        return self.visit_BOOLEAN(type_, **kw)

    def visit_time(self, type_, **kw):
        return self.visit_TIME(type_, **kw)

    def visit_datetime(self, type_, **kw):
        return self.visit_DATETIME(type_, **kw)

    def visit_date(self, type_, **kw):
        return self.visit_DATE(type_, **kw)

    def visit_big_integer(self, type_, **kw):
        return self.visit_BIGINT(type_, **kw)

    def visit_small_integer(self, type_, **kw):
        return self.visit_SMALLINT(type_, **kw)

    def visit_integer(self, type_, **kw):
        return self.visit_INTEGER(type_, **kw)

    def visit_real(self, type_, **kw):
        return self.visit_REAL(type_, **kw)

    def visit_float(self, type_, **kw):
        return self.visit_FLOAT(type_, **kw)

    def visit_double(self, type_, **kw):
        return self.visit_DOUBLE(type_, **kw)

    def visit_numeric(self, type_, **kw):
        return self.visit_NUMERIC(type_, **kw)

    def visit_string(self, type_, **kw):
        return self.visit_VARCHAR(type_, **kw)

    def visit_unicode(self, type_, **kw):
        return self.visit_VARCHAR(type_, **kw)

    def visit_text(self, type_, **kw):
        return self.visit_TEXT(type_, **kw)

    def visit_unicode_text(self, type_, **kw):
        return self.visit_TEXT(type_, **kw)

    def visit_enum(self, type_, **kw):
        return self.visit_VARCHAR(type_, **kw)

    def visit_null(self, type_, **kw):
        raise exc.CompileError("Can't generate DDL for %r; did you forget to specify a type on this Column?" % type_)

    def visit_type_decorator(self, type_, **kw):
        return self.process(type_.type_engine(self.dialect), **kw)

    def visit_user_defined(self, type_, **kw):
        return type_.get_col_spec(**kw)