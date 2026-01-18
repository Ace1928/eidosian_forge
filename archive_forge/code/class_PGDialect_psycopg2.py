from __future__ import annotations
import collections.abc as collections_abc
import logging
import re
from typing import cast
from . import ranges
from ._psycopg_common import _PGDialect_common_psycopg
from ._psycopg_common import _PGExecutionContext_common_psycopg
from .base import PGIdentifierPreparer
from .json import JSON
from .json import JSONB
from ... import types as sqltypes
from ... import util
from ...util import FastIntFlag
from ...util import parse_user_argument_for_enum
class PGDialect_psycopg2(_PGDialect_common_psycopg):
    driver = 'psycopg2'
    supports_statement_cache = True
    supports_server_side_cursors = True
    default_paramstyle = 'pyformat'
    supports_sane_multi_rowcount = False
    execution_ctx_cls = PGExecutionContext_psycopg2
    preparer = PGIdentifierPreparer_psycopg2
    psycopg2_version = (0, 0)
    use_insertmanyvalues_wo_returning = True
    returns_native_bytes = False
    _has_native_hstore = True
    colspecs = util.update_copy(_PGDialect_common_psycopg.colspecs, {JSON: _PGJSON, sqltypes.JSON: _PGJSON, JSONB: _PGJSONB, ranges.INT4RANGE: _Psycopg2NumericRange, ranges.INT8RANGE: _Psycopg2NumericRange, ranges.NUMRANGE: _Psycopg2NumericRange, ranges.DATERANGE: _Psycopg2DateRange, ranges.TSRANGE: _Psycopg2DateTimeRange, ranges.TSTZRANGE: _Psycopg2DateTimeTZRange})

    def __init__(self, executemany_mode='values_only', executemany_batch_page_size=100, **kwargs):
        _PGDialect_common_psycopg.__init__(self, **kwargs)
        if self._native_inet_types:
            raise NotImplementedError('The psycopg2 dialect does not implement ipaddress type handling; native_inet_types cannot be set to ``True`` when using this dialect.')
        self.executemany_mode = parse_user_argument_for_enum(executemany_mode, {EXECUTEMANY_VALUES: ['values_only'], EXECUTEMANY_VALUES_PLUS_BATCH: ['values_plus_batch']}, 'executemany_mode')
        self.executemany_batch_page_size = executemany_batch_page_size
        if self.dbapi and hasattr(self.dbapi, '__version__'):
            m = re.match('(\\d+)\\.(\\d+)(?:\\.(\\d+))?', self.dbapi.__version__)
            if m:
                self.psycopg2_version = tuple((int(x) for x in m.group(1, 2, 3) if x is not None))
            if self.psycopg2_version < (2, 7):
                raise ImportError('psycopg2 version 2.7 or higher is required.')

    def initialize(self, connection):
        super().initialize(connection)
        self._has_native_hstore = self.use_native_hstore and self._hstore_oids(connection.connection.dbapi_connection) is not None
        self.supports_sane_multi_rowcount = self.executemany_mode is not EXECUTEMANY_VALUES_PLUS_BATCH

    @classmethod
    def import_dbapi(cls):
        import psycopg2
        return psycopg2

    @util.memoized_property
    def _psycopg2_extensions(cls):
        from psycopg2 import extensions
        return extensions

    @util.memoized_property
    def _psycopg2_extras(cls):
        from psycopg2 import extras
        return extras

    @util.memoized_property
    def _isolation_lookup(self):
        extensions = self._psycopg2_extensions
        return {'AUTOCOMMIT': extensions.ISOLATION_LEVEL_AUTOCOMMIT, 'READ COMMITTED': extensions.ISOLATION_LEVEL_READ_COMMITTED, 'READ UNCOMMITTED': extensions.ISOLATION_LEVEL_READ_UNCOMMITTED, 'REPEATABLE READ': extensions.ISOLATION_LEVEL_REPEATABLE_READ, 'SERIALIZABLE': extensions.ISOLATION_LEVEL_SERIALIZABLE}

    def set_isolation_level(self, dbapi_connection, level):
        dbapi_connection.set_isolation_level(self._isolation_lookup[level])

    def set_readonly(self, connection, value):
        connection.readonly = value

    def get_readonly(self, connection):
        return connection.readonly

    def set_deferrable(self, connection, value):
        connection.deferrable = value

    def get_deferrable(self, connection):
        return connection.deferrable

    def on_connect(self):
        extras = self._psycopg2_extras
        fns = []
        if self.client_encoding is not None:

            def on_connect(dbapi_conn):
                dbapi_conn.set_client_encoding(self.client_encoding)
            fns.append(on_connect)
        if self.dbapi:

            def on_connect(dbapi_conn):
                extras.register_uuid(None, dbapi_conn)
            fns.append(on_connect)
        if self.dbapi and self.use_native_hstore:

            def on_connect(dbapi_conn):
                hstore_oids = self._hstore_oids(dbapi_conn)
                if hstore_oids is not None:
                    oid, array_oid = hstore_oids
                    kw = {'oid': oid}
                    kw['array_oid'] = array_oid
                    extras.register_hstore(dbapi_conn, **kw)
            fns.append(on_connect)
        if self.dbapi and self._json_deserializer:

            def on_connect(dbapi_conn):
                extras.register_default_json(dbapi_conn, loads=self._json_deserializer)
                extras.register_default_jsonb(dbapi_conn, loads=self._json_deserializer)
            fns.append(on_connect)
        if fns:

            def on_connect(dbapi_conn):
                for fn in fns:
                    fn(dbapi_conn)
            return on_connect
        else:
            return None

    def do_executemany(self, cursor, statement, parameters, context=None):
        if self.executemany_mode is EXECUTEMANY_VALUES_PLUS_BATCH:
            if self.executemany_batch_page_size:
                kwargs = {'page_size': self.executemany_batch_page_size}
            else:
                kwargs = {}
            self._psycopg2_extras.execute_batch(cursor, statement, parameters, **kwargs)
        else:
            cursor.executemany(statement, parameters)

    def do_begin_twophase(self, connection, xid):
        connection.connection.tpc_begin(xid)

    def do_prepare_twophase(self, connection, xid):
        connection.connection.tpc_prepare()

    def _do_twophase(self, dbapi_conn, operation, xid, recover=False):
        if recover:
            if dbapi_conn.status != self._psycopg2_extensions.STATUS_READY:
                dbapi_conn.rollback()
            operation(xid)
        else:
            operation()

    def do_rollback_twophase(self, connection, xid, is_prepared=True, recover=False):
        dbapi_conn = connection.connection.dbapi_connection
        self._do_twophase(dbapi_conn, dbapi_conn.tpc_rollback, xid, recover=recover)

    def do_commit_twophase(self, connection, xid, is_prepared=True, recover=False):
        dbapi_conn = connection.connection.dbapi_connection
        self._do_twophase(dbapi_conn, dbapi_conn.tpc_commit, xid, recover=recover)

    @util.memoized_instancemethod
    def _hstore_oids(self, dbapi_connection):
        extras = self._psycopg2_extras
        oids = extras.HstoreAdapter.get_oids(dbapi_connection)
        if oids is not None and oids[0]:
            return oids[0:2]
        else:
            return None

    def is_disconnect(self, e, connection, cursor):
        if isinstance(e, self.dbapi.Error):
            if getattr(connection, 'closed', False):
                return True
            str_e = str(e).partition('\n')[0]
            for msg in ['terminating connection', 'closed the connection', 'connection not open', 'could not receive data from server', 'could not send data to server', 'connection already closed', 'cursor already closed', 'losed the connection unexpectedly', 'connection has been closed unexpectedly', 'SSL error: decryption failed or bad record mac', 'SSL SYSCALL error: Bad file descriptor', 'SSL SYSCALL error: EOF detected', 'SSL SYSCALL error: Operation timed out', 'SSL SYSCALL error: Bad address']:
                idx = str_e.find(msg)
                if idx >= 0 and '"' not in str_e[:idx]:
                    return True
        return False