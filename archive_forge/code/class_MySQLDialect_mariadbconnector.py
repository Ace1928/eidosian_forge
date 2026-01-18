import re
from uuid import UUID as _python_UUID
from .base import MySQLCompiler
from .base import MySQLDialect
from .base import MySQLExecutionContext
from ... import sql
from ... import util
from ...sql import sqltypes
class MySQLDialect_mariadbconnector(MySQLDialect):
    driver = 'mariadbconnector'
    supports_statement_cache = True
    supports_unicode_statements = True
    encoding = 'utf8mb4'
    convert_unicode = True
    supports_sane_rowcount = True
    supports_sane_multi_rowcount = True
    supports_native_decimal = True
    default_paramstyle = 'qmark'
    execution_ctx_cls = MySQLExecutionContext_mariadbconnector
    statement_compiler = MySQLCompiler_mariadbconnector
    supports_server_side_cursors = True
    colspecs = util.update_copy(MySQLDialect.colspecs, {sqltypes.Uuid: _MariaDBUUID})

    @util.memoized_property
    def _dbapi_version(self):
        if self.dbapi and hasattr(self.dbapi, '__version__'):
            return tuple([int(x) for x in re.findall('(\\d+)(?:[-\\.]?|$)', self.dbapi.__version__)])
        else:
            return (99, 99, 99)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.paramstyle = 'qmark'
        if self.dbapi is not None:
            if self._dbapi_version < mariadb_cpy_minimum_version:
                raise NotImplementedError('The minimum required version for MariaDB Connector/Python is %s' % '.'.join((str(x) for x in mariadb_cpy_minimum_version)))

    @classmethod
    def import_dbapi(cls):
        return __import__('mariadb')

    def is_disconnect(self, e, connection, cursor):
        if super().is_disconnect(e, connection, cursor):
            return True
        elif isinstance(e, self.dbapi.Error):
            str_e = str(e).lower()
            return 'not connected' in str_e or "isn't valid" in str_e
        else:
            return False

    def create_connect_args(self, url):
        opts = url.translate_connect_args()
        int_params = ['connect_timeout', 'read_timeout', 'write_timeout', 'client_flag', 'port', 'pool_size']
        bool_params = ['local_infile', 'ssl_verify_cert', 'ssl', 'pool_reset_connection']
        for key in int_params:
            util.coerce_kw_type(opts, key, int)
        for key in bool_params:
            util.coerce_kw_type(opts, key, bool)
        client_flag = opts.get('client_flag', 0)
        if self.dbapi is not None:
            try:
                CLIENT_FLAGS = __import__(self.dbapi.__name__ + '.constants.CLIENT').constants.CLIENT
                client_flag |= CLIENT_FLAGS.FOUND_ROWS
            except (AttributeError, ImportError):
                self.supports_sane_rowcount = False
            opts['client_flag'] = client_flag
        return [[], opts]

    def _extract_error_code(self, exception):
        try:
            rc = exception.errno
        except:
            rc = -1
        return rc

    def _detect_charset(self, connection):
        return 'utf8mb4'

    def get_isolation_level_values(self, dbapi_connection):
        return ('SERIALIZABLE', 'READ UNCOMMITTED', 'READ COMMITTED', 'REPEATABLE READ', 'AUTOCOMMIT')

    def set_isolation_level(self, connection, level):
        if level == 'AUTOCOMMIT':
            connection.autocommit = True
        else:
            connection.autocommit = False
            super().set_isolation_level(connection, level)

    def do_begin_twophase(self, connection, xid):
        connection.execute(sql.text('XA BEGIN :xid').bindparams(sql.bindparam('xid', xid, literal_execute=True)))

    def do_prepare_twophase(self, connection, xid):
        connection.execute(sql.text('XA END :xid').bindparams(sql.bindparam('xid', xid, literal_execute=True)))
        connection.execute(sql.text('XA PREPARE :xid').bindparams(sql.bindparam('xid', xid, literal_execute=True)))

    def do_rollback_twophase(self, connection, xid, is_prepared=True, recover=False):
        if not is_prepared:
            connection.execute(sql.text('XA END :xid').bindparams(sql.bindparam('xid', xid, literal_execute=True)))
        connection.execute(sql.text('XA ROLLBACK :xid').bindparams(sql.bindparam('xid', xid, literal_execute=True)))

    def do_commit_twophase(self, connection, xid, is_prepared=True, recover=False):
        if not is_prepared:
            self.do_prepare_twophase(connection, xid)
        connection.execute(sql.text('XA COMMIT :xid').bindparams(sql.bindparam('xid', xid, literal_execute=True)))