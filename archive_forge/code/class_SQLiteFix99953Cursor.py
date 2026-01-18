from the connection pool, such as when using an ORM :class:`.Session` where
from working correctly.  The pysqlite DBAPI driver has several
import math
import os
import re
from .base import DATE
from .base import DATETIME
from .base import SQLiteDialect
from ... import exc
from ... import pool
from ... import types as sqltypes
from ... import util
class SQLiteFix99953Cursor(sqlite3.Cursor):

    def execute(self, sql, parameters=()):
        _test_sql(sql)
        if first_bind in sql:
            parameters = _numeric_param_as_dict(parameters)
        return super().execute(sql, parameters)

    def executemany(self, sql, parameters):
        _test_sql(sql)
        if first_bind in sql:
            parameters = [_numeric_param_as_dict(p) for p in parameters]
        return super().executemany(sql, parameters)