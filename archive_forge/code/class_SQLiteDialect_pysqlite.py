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
class SQLiteDialect_pysqlite(SQLiteDialect):
    default_paramstyle = 'qmark'
    supports_statement_cache = True
    returns_native_bytes = True
    colspecs = util.update_copy(SQLiteDialect.colspecs, {sqltypes.Date: _SQLite_pysqliteDate, sqltypes.TIMESTAMP: _SQLite_pysqliteTimeStamp})
    description_encoding = None
    driver = 'pysqlite'

    @classmethod
    def import_dbapi(cls):
        from sqlite3 import dbapi2 as sqlite
        return sqlite

    @classmethod
    def _is_url_file_db(cls, url):
        if (url.database and url.database != ':memory:') and url.query.get('mode', None) != 'memory':
            return True
        else:
            return False

    @classmethod
    def get_pool_class(cls, url):
        if cls._is_url_file_db(url):
            return pool.QueuePool
        else:
            return pool.SingletonThreadPool

    def _get_server_version_info(self, connection):
        return self.dbapi.sqlite_version_info
    _isolation_lookup = SQLiteDialect._isolation_lookup.union({'AUTOCOMMIT': None})

    def set_isolation_level(self, dbapi_connection, level):
        if level == 'AUTOCOMMIT':
            dbapi_connection.isolation_level = None
        else:
            dbapi_connection.isolation_level = ''
            return super().set_isolation_level(dbapi_connection, level)

    def on_connect(self):

        def regexp(a, b):
            if b is None:
                return None
            return re.search(a, b) is not None
        if util.py38 and self._get_server_version_info(None) >= (3, 9):
            create_func_kw = {'deterministic': True}
        else:
            create_func_kw = {}

        def set_regexp(dbapi_connection):
            dbapi_connection.create_function('regexp', 2, regexp, **create_func_kw)

        def floor_func(dbapi_connection):
            dbapi_connection.create_function('floor', 1, math.floor, **create_func_kw)
        fns = [set_regexp, floor_func]

        def connect(conn):
            for fn in fns:
                fn(conn)
        return connect

    def create_connect_args(self, url):
        if url.username or url.password or url.host or url.port:
            raise exc.ArgumentError('Invalid SQLite URL: %s\nValid SQLite URL forms are:\n sqlite:///:memory: (or, sqlite://)\n sqlite:///relative/path/to/file.db\n sqlite:////absolute/path/to/file.db' % (url,))
        pysqlite_args = [('uri', bool), ('timeout', float), ('isolation_level', str), ('detect_types', int), ('check_same_thread', bool), ('cached_statements', int)]
        opts = url.query
        pysqlite_opts = {}
        for key, type_ in pysqlite_args:
            util.coerce_kw_type(opts, key, type_, dest=pysqlite_opts)
        if pysqlite_opts.get('uri', False):
            uri_opts = dict(opts)
            for key, type_ in pysqlite_args:
                uri_opts.pop(key, None)
            filename = url.database
            if uri_opts:
                filename += '?' + '&'.join(('%s=%s' % (key, uri_opts[key]) for key in sorted(uri_opts)))
        else:
            filename = url.database or ':memory:'
            if filename != ':memory:':
                filename = os.path.abspath(filename)
        pysqlite_opts.setdefault('check_same_thread', not self._is_url_file_db(url))
        return ([filename], pysqlite_opts)

    def is_disconnect(self, e, connection, cursor):
        return isinstance(e, self.dbapi.ProgrammingError) and 'Cannot operate on a closed database.' in str(e)