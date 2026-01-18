from __future__ import annotations
import collections.abc as collections_abc
import datetime as dt
import decimal
import enum
import json
import pickle
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from uuid import UUID as _python_UUID
from . import coercions
from . import elements
from . import operators
from . import roles
from . import type_api
from .base import _NONE_NAME
from .base import NO_ARG
from .base import SchemaEventTarget
from .cache_key import HasCacheKey
from .elements import quoted_name
from .elements import Slice
from .elements import TypeCoerce as type_coerce  # noqa
from .type_api import Emulated
from .type_api import NativeForEmulated  # noqa
from .type_api import to_instance as to_instance
from .type_api import TypeDecorator as TypeDecorator
from .type_api import TypeEngine as TypeEngine
from .type_api import TypeEngineMixin
from .type_api import Variant  # noqa
from .visitors import InternalTraversal
from .. import event
from .. import exc
from .. import inspection
from .. import util
from ..engine import processors
from ..util import langhelpers
from ..util import OrderedDict
from ..util.typing import is_literal
from ..util.typing import Literal
from ..util.typing import typing_get_args
class _Binary(TypeEngine[bytes]):
    """Define base behavior for binary types."""

    def __init__(self, length: Optional[int]=None):
        self.length = length

    def literal_processor(self, dialect):

        def process(value):
            value = value.decode(dialect._legacy_binary_type_literal_encoding).replace("'", "''")
            return "'%s'" % value
        return process

    @property
    def python_type(self):
        return bytes

    def bind_processor(self, dialect):
        if dialect.dbapi is None:
            return None
        DBAPIBinary = dialect.dbapi.Binary

        def process(value):
            if value is not None:
                return DBAPIBinary(value)
            else:
                return None
        return process

    def result_processor(self, dialect, coltype):
        if dialect.returns_native_bytes:
            return None

        def process(value):
            if value is not None:
                value = bytes(value)
            return value
        return process

    def coerce_compared_value(self, op, value):
        """See :meth:`.TypeEngine.coerce_compared_value` for a description."""
        if isinstance(value, str):
            return self
        else:
            return super().coerce_compared_value(op, value)

    def get_dbapi_type(self, dbapi):
        return dbapi.BINARY