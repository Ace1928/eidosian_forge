from __future__ import annotations
import codecs
import datetime
import operator
import re
from typing import overload
from typing import TYPE_CHECKING
from uuid import UUID as _python_UUID
from . import information_schema as ischema
from .json import JSON
from .json import JSONIndexType
from .json import JSONPathType
from ... import exc
from ... import Identity
from ... import schema as sa_schema
from ... import Sequence
from ... import sql
from ... import text
from ... import util
from ...engine import cursor as _cursor
from ...engine import default
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import coercions
from ...sql import compiler
from ...sql import elements
from ...sql import expression
from ...sql import func
from ...sql import quoted_name
from ...sql import roles
from ...sql import sqltypes
from ...sql import try_cast as try_cast  # noqa: F401
from ...sql import util as sql_util
from ...sql._typing import is_sql_compiler
from ...sql.compiler import InsertmanyvaluesSentinelOpts
from ...sql.elements import TryCast as TryCast  # noqa: F401
from ...types import BIGINT
from ...types import BINARY
from ...types import CHAR
from ...types import DATE
from ...types import DATETIME
from ...types import DECIMAL
from ...types import FLOAT
from ...types import INTEGER
from ...types import NCHAR
from ...types import NUMERIC
from ...types import NVARCHAR
from ...types import SMALLINT
from ...types import TEXT
from ...types import VARCHAR
from ...util import update_wrapper
from ...util.typing import Literal
from
from
class MSTypeCompiler(compiler.GenericTypeCompiler):

    def _extend(self, spec, type_, length=None):
        """Extend a string-type declaration with standard SQL
        COLLATE annotations.

        """
        if getattr(type_, 'collation', None):
            collation = 'COLLATE %s' % type_.collation
        else:
            collation = None
        if not length:
            length = type_.length
        if length:
            spec = spec + '(%s)' % length
        return ' '.join([c for c in (spec, collation) if c is not None])

    def visit_double(self, type_, **kw):
        return self.visit_DOUBLE_PRECISION(type_, **kw)

    def visit_FLOAT(self, type_, **kw):
        precision = getattr(type_, 'precision', None)
        if precision is None:
            return 'FLOAT'
        else:
            return 'FLOAT(%(precision)s)' % {'precision': precision}

    def visit_TINYINT(self, type_, **kw):
        return 'TINYINT'

    def visit_TIME(self, type_, **kw):
        precision = getattr(type_, 'precision', None)
        if precision is not None:
            return 'TIME(%s)' % precision
        else:
            return 'TIME'

    def visit_TIMESTAMP(self, type_, **kw):
        return 'TIMESTAMP'

    def visit_ROWVERSION(self, type_, **kw):
        return 'ROWVERSION'

    def visit_datetime(self, type_, **kw):
        if type_.timezone:
            return self.visit_DATETIMEOFFSET(type_, **kw)
        else:
            return self.visit_DATETIME(type_, **kw)

    def visit_DATETIMEOFFSET(self, type_, **kw):
        precision = getattr(type_, 'precision', None)
        if precision is not None:
            return 'DATETIMEOFFSET(%s)' % type_.precision
        else:
            return 'DATETIMEOFFSET'

    def visit_DATETIME2(self, type_, **kw):
        precision = getattr(type_, 'precision', None)
        if precision is not None:
            return 'DATETIME2(%s)' % precision
        else:
            return 'DATETIME2'

    def visit_SMALLDATETIME(self, type_, **kw):
        return 'SMALLDATETIME'

    def visit_unicode(self, type_, **kw):
        return self.visit_NVARCHAR(type_, **kw)

    def visit_text(self, type_, **kw):
        if self.dialect.deprecate_large_types:
            return self.visit_VARCHAR(type_, **kw)
        else:
            return self.visit_TEXT(type_, **kw)

    def visit_unicode_text(self, type_, **kw):
        if self.dialect.deprecate_large_types:
            return self.visit_NVARCHAR(type_, **kw)
        else:
            return self.visit_NTEXT(type_, **kw)

    def visit_NTEXT(self, type_, **kw):
        return self._extend('NTEXT', type_)

    def visit_TEXT(self, type_, **kw):
        return self._extend('TEXT', type_)

    def visit_VARCHAR(self, type_, **kw):
        return self._extend('VARCHAR', type_, length=type_.length or 'max')

    def visit_CHAR(self, type_, **kw):
        return self._extend('CHAR', type_)

    def visit_NCHAR(self, type_, **kw):
        return self._extend('NCHAR', type_)

    def visit_NVARCHAR(self, type_, **kw):
        return self._extend('NVARCHAR', type_, length=type_.length or 'max')

    def visit_date(self, type_, **kw):
        if self.dialect.server_version_info < MS_2008_VERSION:
            return self.visit_DATETIME(type_, **kw)
        else:
            return self.visit_DATE(type_, **kw)

    def visit__BASETIMEIMPL(self, type_, **kw):
        return self.visit_time(type_, **kw)

    def visit_time(self, type_, **kw):
        if self.dialect.server_version_info < MS_2008_VERSION:
            return self.visit_DATETIME(type_, **kw)
        else:
            return self.visit_TIME(type_, **kw)

    def visit_large_binary(self, type_, **kw):
        if self.dialect.deprecate_large_types:
            return self.visit_VARBINARY(type_, **kw)
        else:
            return self.visit_IMAGE(type_, **kw)

    def visit_IMAGE(self, type_, **kw):
        return 'IMAGE'

    def visit_XML(self, type_, **kw):
        return 'XML'

    def visit_VARBINARY(self, type_, **kw):
        text = self._extend('VARBINARY', type_, length=type_.length or 'max')
        if getattr(type_, 'filestream', False):
            text += ' FILESTREAM'
        return text

    def visit_boolean(self, type_, **kw):
        return self.visit_BIT(type_)

    def visit_BIT(self, type_, **kw):
        return 'BIT'

    def visit_JSON(self, type_, **kw):
        return self._extend('NVARCHAR', type_, length='max')

    def visit_MONEY(self, type_, **kw):
        return 'MONEY'

    def visit_SMALLMONEY(self, type_, **kw):
        return 'SMALLMONEY'

    def visit_uuid(self, type_, **kw):
        if type_.native_uuid:
            return self.visit_UNIQUEIDENTIFIER(type_, **kw)
        else:
            return super().visit_uuid(type_, **kw)

    def visit_UNIQUEIDENTIFIER(self, type_, **kw):
        return 'UNIQUEIDENTIFIER'

    def visit_SQL_VARIANT(self, type_, **kw):
        return 'SQL_VARIANT'