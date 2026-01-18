from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
from functools import wraps
import re
from . import dictionary
from .types import _OracleBoolean
from .types import _OracleDate
from .types import BFILE
from .types import BINARY_DOUBLE
from .types import BINARY_FLOAT
from .types import DATE
from .types import FLOAT
from .types import INTERVAL
from .types import LONG
from .types import NCLOB
from .types import NUMBER
from .types import NVARCHAR2  # noqa
from .types import OracleRaw  # noqa
from .types import RAW
from .types import ROWID  # noqa
from .types import TIMESTAMP
from .types import VARCHAR2  # noqa
from ... import Computed
from ... import exc
from ... import schema as sa_schema
from ... import sql
from ... import util
from ...engine import default
from ...engine import ObjectKind
from ...engine import ObjectScope
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import and_
from ...sql import bindparam
from ...sql import compiler
from ...sql import expression
from ...sql import func
from ...sql import null
from ...sql import or_
from ...sql import select
from ...sql import sqltypes
from ...sql import util as sql_util
from ...sql import visitors
from ...sql.visitors import InternalTraversal
from ...types import BLOB
from ...types import CHAR
from ...types import CLOB
from ...types import DOUBLE_PRECISION
from ...types import INTEGER
from ...types import NCHAR
from ...types import NVARCHAR
from ...types import REAL
from ...types import VARCHAR
class OracleTypeCompiler(compiler.GenericTypeCompiler):

    def visit_datetime(self, type_, **kw):
        return self.visit_DATE(type_, **kw)

    def visit_float(self, type_, **kw):
        return self.visit_FLOAT(type_, **kw)

    def visit_double(self, type_, **kw):
        return self.visit_DOUBLE_PRECISION(type_, **kw)

    def visit_unicode(self, type_, **kw):
        if self.dialect._use_nchar_for_unicode:
            return self.visit_NVARCHAR2(type_, **kw)
        else:
            return self.visit_VARCHAR2(type_, **kw)

    def visit_INTERVAL(self, type_, **kw):
        return 'INTERVAL DAY%s TO SECOND%s' % (type_.day_precision is not None and '(%d)' % type_.day_precision or '', type_.second_precision is not None and '(%d)' % type_.second_precision or '')

    def visit_LONG(self, type_, **kw):
        return 'LONG'

    def visit_TIMESTAMP(self, type_, **kw):
        if getattr(type_, 'local_timezone', False):
            return 'TIMESTAMP WITH LOCAL TIME ZONE'
        elif type_.timezone:
            return 'TIMESTAMP WITH TIME ZONE'
        else:
            return 'TIMESTAMP'

    def visit_DOUBLE_PRECISION(self, type_, **kw):
        return self._generate_numeric(type_, 'DOUBLE PRECISION', **kw)

    def visit_BINARY_DOUBLE(self, type_, **kw):
        return self._generate_numeric(type_, 'BINARY_DOUBLE', **kw)

    def visit_BINARY_FLOAT(self, type_, **kw):
        return self._generate_numeric(type_, 'BINARY_FLOAT', **kw)

    def visit_FLOAT(self, type_, **kw):
        kw['_requires_binary_precision'] = True
        return self._generate_numeric(type_, 'FLOAT', **kw)

    def visit_NUMBER(self, type_, **kw):
        return self._generate_numeric(type_, 'NUMBER', **kw)

    def _generate_numeric(self, type_, name, precision=None, scale=None, _requires_binary_precision=False, **kw):
        if precision is None:
            precision = getattr(type_, 'precision', None)
        if _requires_binary_precision:
            binary_precision = getattr(type_, 'binary_precision', None)
            if precision and binary_precision is None:
                estimated_binary_precision = int(precision / 0.30103)
                raise exc.ArgumentError(f"Oracle FLOAT types use 'binary precision', which does not convert cleanly from decimal 'precision'.  Please specify this type with a separate Oracle variant, such as {type_.__class__.__name__}(precision={precision}).with_variant(oracle.FLOAT(binary_precision={estimated_binary_precision}), 'oracle'), so that the Oracle specific 'binary_precision' may be specified accurately.")
            else:
                precision = binary_precision
        if scale is None:
            scale = getattr(type_, 'scale', None)
        if precision is None:
            return name
        elif scale is None:
            n = '%(name)s(%(precision)s)'
            return n % {'name': name, 'precision': precision}
        else:
            n = '%(name)s(%(precision)s, %(scale)s)'
            return n % {'name': name, 'precision': precision, 'scale': scale}

    def visit_string(self, type_, **kw):
        return self.visit_VARCHAR2(type_, **kw)

    def visit_VARCHAR2(self, type_, **kw):
        return self._visit_varchar(type_, '', '2')

    def visit_NVARCHAR2(self, type_, **kw):
        return self._visit_varchar(type_, 'N', '2')
    visit_NVARCHAR = visit_NVARCHAR2

    def visit_VARCHAR(self, type_, **kw):
        return self._visit_varchar(type_, '', '')

    def _visit_varchar(self, type_, n, num):
        if not type_.length:
            return '%(n)sVARCHAR%(two)s' % {'two': num, 'n': n}
        elif not n and self.dialect._supports_char_length:
            varchar = 'VARCHAR%(two)s(%(length)s CHAR)'
            return varchar % {'length': type_.length, 'two': num}
        else:
            varchar = '%(n)sVARCHAR%(two)s(%(length)s)'
            return varchar % {'length': type_.length, 'two': num, 'n': n}

    def visit_text(self, type_, **kw):
        return self.visit_CLOB(type_, **kw)

    def visit_unicode_text(self, type_, **kw):
        if self.dialect._use_nchar_for_unicode:
            return self.visit_NCLOB(type_, **kw)
        else:
            return self.visit_CLOB(type_, **kw)

    def visit_large_binary(self, type_, **kw):
        return self.visit_BLOB(type_, **kw)

    def visit_big_integer(self, type_, **kw):
        return self.visit_NUMBER(type_, precision=19, **kw)

    def visit_boolean(self, type_, **kw):
        return self.visit_SMALLINT(type_, **kw)

    def visit_RAW(self, type_, **kw):
        if type_.length:
            return 'RAW(%(length)s)' % {'length': type_.length}
        else:
            return 'RAW'

    def visit_ROWID(self, type_, **kw):
        return 'ROWID'