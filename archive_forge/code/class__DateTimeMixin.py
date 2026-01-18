from the proposed insertion. These values are specified using the
from __future__ import annotations
import datetime
import numbers
import re
from typing import Optional
from .json import JSON
from .json import JSONIndexType
from .json import JSONPathType
from ... import exc
from ... import schema as sa_schema
from ... import sql
from ... import text
from ... import types as sqltypes
from ... import util
from ...engine import default
from ...engine import processors
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import coercions
from ...sql import ColumnElement
from ...sql import compiler
from ...sql import elements
from ...sql import roles
from ...sql import schema
from ...types import BLOB  # noqa
from ...types import BOOLEAN  # noqa
from ...types import CHAR  # noqa
from ...types import DECIMAL  # noqa
from ...types import FLOAT  # noqa
from ...types import INTEGER  # noqa
from ...types import NUMERIC  # noqa
from ...types import REAL  # noqa
from ...types import SMALLINT  # noqa
from ...types import TEXT  # noqa
from ...types import TIMESTAMP  # noqa
from ...types import VARCHAR  # noqa
class _DateTimeMixin:
    _reg = None
    _storage_format = None

    def __init__(self, storage_format=None, regexp=None, **kw):
        super().__init__(**kw)
        if regexp is not None:
            self._reg = re.compile(regexp)
        if storage_format is not None:
            self._storage_format = storage_format

    @property
    def format_is_text_affinity(self):
        """return True if the storage format will automatically imply
        a TEXT affinity.

        If the storage format contains no non-numeric characters,
        it will imply a NUMERIC storage format on SQLite; in this case,
        the type will generate its DDL as DATE_CHAR, DATETIME_CHAR,
        TIME_CHAR.

        """
        spec = self._storage_format % {'year': 0, 'month': 0, 'day': 0, 'hour': 0, 'minute': 0, 'second': 0, 'microsecond': 0}
        return bool(re.search('[^0-9]', spec))

    def adapt(self, cls, **kw):
        if issubclass(cls, _DateTimeMixin):
            if self._storage_format:
                kw['storage_format'] = self._storage_format
            if self._reg:
                kw['regexp'] = self._reg
        return super().adapt(cls, **kw)

    def literal_processor(self, dialect):
        bp = self.bind_processor(dialect)

        def process(value):
            return "'%s'" % bp(value)
        return process