import decimal
import re
from . import ranges
from .array import ARRAY as PGARRAY
from .base import _DECIMAL_TYPES
from .base import _FLOAT_TYPES
from .base import _INT_TYPES
from .base import ENUM
from .base import INTERVAL
from .base import PGCompiler
from .base import PGDialect
from .base import PGExecutionContext
from .base import PGIdentifierPreparer
from .json import JSON
from .json import JSONB
from .json import JSONPathType
from .pg_catalog import _SpaceVector
from .pg_catalog import OIDVECTOR
from .types import CITEXT
from ... import exc
from ... import util
from ...engine import processors
from ...sql import sqltypes
from ...sql.elements import quoted_name
class PGDialect_pg8000(PGDialect):
    driver = 'pg8000'
    supports_statement_cache = True
    supports_unicode_statements = True
    supports_unicode_binds = True
    default_paramstyle = 'format'
    supports_sane_multi_rowcount = True
    execution_ctx_cls = PGExecutionContext_pg8000
    statement_compiler = PGCompiler_pg8000
    preparer = PGIdentifierPreparer_pg8000
    supports_server_side_cursors = True
    render_bind_cast = True
    description_encoding = None
    colspecs = util.update_copy(PGDialect.colspecs, {sqltypes.String: _PGString, sqltypes.Numeric: _PGNumericNoBind, sqltypes.Float: _PGFloat, sqltypes.JSON: _PGJSON, sqltypes.Boolean: _PGBoolean, sqltypes.NullType: _PGNullType, JSONB: _PGJSONB, CITEXT: CITEXT, sqltypes.JSON.JSONPathType: _PGJSONPathType, sqltypes.JSON.JSONIndexType: _PGJSONIndexType, sqltypes.JSON.JSONIntIndexType: _PGJSONIntIndexType, sqltypes.JSON.JSONStrIndexType: _PGJSONStrIndexType, sqltypes.Interval: _PGInterval, INTERVAL: _PGInterval, sqltypes.DateTime: _PGTimeStamp, sqltypes.DateTime: _PGTimeStamp, sqltypes.Date: _PGDate, sqltypes.Time: _PGTime, sqltypes.Integer: _PGInteger, sqltypes.SmallInteger: _PGSmallInteger, sqltypes.BigInteger: _PGBigInteger, sqltypes.Enum: _PGEnum, sqltypes.ARRAY: _PGARRAY, OIDVECTOR: _PGOIDVECTOR, ranges.INT4RANGE: _Pg8000Range, ranges.INT8RANGE: _Pg8000Range, ranges.NUMRANGE: _Pg8000Range, ranges.DATERANGE: _Pg8000Range, ranges.TSRANGE: _Pg8000Range, ranges.TSTZRANGE: _Pg8000Range, ranges.INT4MULTIRANGE: _Pg8000MultiRange, ranges.INT8MULTIRANGE: _Pg8000MultiRange, ranges.NUMMULTIRANGE: _Pg8000MultiRange, ranges.DATEMULTIRANGE: _Pg8000MultiRange, ranges.TSMULTIRANGE: _Pg8000MultiRange, ranges.TSTZMULTIRANGE: _Pg8000MultiRange})

    def __init__(self, client_encoding=None, **kwargs):
        PGDialect.__init__(self, **kwargs)
        self.client_encoding = client_encoding
        if self._dbapi_version < (1, 16, 6):
            raise NotImplementedError('pg8000 1.16.6 or greater is required')
        if self._native_inet_types:
            raise NotImplementedError('The pg8000 dialect does not fully implement ipaddress type handling; INET is supported by default, CIDR is not')

    @util.memoized_property
    def _dbapi_version(self):
        if self.dbapi and hasattr(self.dbapi, '__version__'):
            return tuple([int(x) for x in re.findall('(\\d+)(?:[-\\.]?|$)', self.dbapi.__version__)])
        else:
            return (99, 99, 99)

    @classmethod
    def import_dbapi(cls):
        return __import__('pg8000')

    def create_connect_args(self, url):
        opts = url.translate_connect_args(username='user')
        if 'port' in opts:
            opts['port'] = int(opts['port'])
        opts.update(url.query)
        return ([], opts)

    def is_disconnect(self, e, connection, cursor):
        if isinstance(e, self.dbapi.InterfaceError) and 'network error' in str(e):
            return True
        return 'connection is closed' in str(e)

    def get_isolation_level_values(self, dbapi_connection):
        return ('AUTOCOMMIT', 'READ COMMITTED', 'READ UNCOMMITTED', 'REPEATABLE READ', 'SERIALIZABLE')

    def set_isolation_level(self, dbapi_connection, level):
        level = level.replace('_', ' ')
        if level == 'AUTOCOMMIT':
            dbapi_connection.autocommit = True
        else:
            dbapi_connection.autocommit = False
            cursor = dbapi_connection.cursor()
            cursor.execute(f'SET SESSION CHARACTERISTICS AS TRANSACTION ISOLATION LEVEL {level}')
            cursor.execute('COMMIT')
            cursor.close()

    def set_readonly(self, connection, value):
        cursor = connection.cursor()
        try:
            cursor.execute('SET SESSION CHARACTERISTICS AS TRANSACTION %s' % ('READ ONLY' if value else 'READ WRITE'))
            cursor.execute('COMMIT')
        finally:
            cursor.close()

    def get_readonly(self, connection):
        cursor = connection.cursor()
        try:
            cursor.execute('show transaction_read_only')
            val = cursor.fetchone()[0]
        finally:
            cursor.close()
        return val == 'on'

    def set_deferrable(self, connection, value):
        cursor = connection.cursor()
        try:
            cursor.execute('SET SESSION CHARACTERISTICS AS TRANSACTION %s' % ('DEFERRABLE' if value else 'NOT DEFERRABLE'))
            cursor.execute('COMMIT')
        finally:
            cursor.close()

    def get_deferrable(self, connection):
        cursor = connection.cursor()
        try:
            cursor.execute('show transaction_deferrable')
            val = cursor.fetchone()[0]
        finally:
            cursor.close()
        return val == 'on'

    def _set_client_encoding(self, dbapi_connection, client_encoding):
        cursor = dbapi_connection.cursor()
        cursor.execute(f"SET CLIENT_ENCODING TO '{client_encoding.replace("'", "''")}'")
        cursor.execute('COMMIT')
        cursor.close()

    def do_begin_twophase(self, connection, xid):
        connection.connection.tpc_begin((0, xid, ''))

    def do_prepare_twophase(self, connection, xid):
        connection.connection.tpc_prepare()

    def do_rollback_twophase(self, connection, xid, is_prepared=True, recover=False):
        connection.connection.tpc_rollback((0, xid, ''))

    def do_commit_twophase(self, connection, xid, is_prepared=True, recover=False):
        connection.connection.tpc_commit((0, xid, ''))

    def do_recover_twophase(self, connection):
        return [row[1] for row in connection.connection.tpc_recover()]

    def on_connect(self):
        fns = []

        def on_connect(conn):
            conn.py_types[quoted_name] = conn.py_types[str]
        fns.append(on_connect)
        if self.client_encoding is not None:

            def on_connect(conn):
                self._set_client_encoding(conn, self.client_encoding)
            fns.append(on_connect)
        if self._native_inet_types is False:

            def on_connect(conn):
                conn.register_in_adapter(869, lambda s: s)
                conn.register_in_adapter(650, lambda s: s)
            fns.append(on_connect)
        if self._json_deserializer:

            def on_connect(conn):
                conn.register_in_adapter(114, self._json_deserializer)
                conn.register_in_adapter(3802, self._json_deserializer)
            fns.append(on_connect)
        if len(fns) > 0:

            def on_connect(conn):
                for fn in fns:
                    fn(conn)
            return on_connect
        else:
            return None

    @util.memoized_property
    def _dialect_specific_select_one(self):
        return ';'