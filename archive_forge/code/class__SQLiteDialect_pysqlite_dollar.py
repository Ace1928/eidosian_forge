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
class _SQLiteDialect_pysqlite_dollar(_SQLiteDialect_pysqlite_numeric):
    """numeric dialect that uses $ for testing only

    internal use only.  This dialect is **NOT** supported by SQLAlchemy
    and may change at any time.

    """
    supports_statement_cache = True
    default_paramstyle = 'numeric_dollar'
    driver = 'pysqlite_dollar'
    _first_bind = '$1'
    _not_in_statement_regexp = re.compile('[^\\d]:\\d+')

    def __init__(self, *arg, **kw):
        kw.setdefault('paramstyle', 'numeric_dollar')
        super().__init__(*arg, **kw)