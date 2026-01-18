import re
from .base import MySQLCompiler
from .base import MySQLDialect
from .base import MySQLExecutionContext
from .base import MySQLIdentifierPreparer
from .base import TEXT
from ... import sql
from ... import util
class MySQLDialect_mysqldb(MySQLDialect):
    driver = 'mysqldb'
    supports_statement_cache = True
    supports_unicode_statements = True
    supports_sane_rowcount = True
    supports_sane_multi_rowcount = True
    supports_native_decimal = True
    default_paramstyle = 'format'
    execution_ctx_cls = MySQLExecutionContext_mysqldb
    statement_compiler = MySQLCompiler_mysqldb
    preparer = MySQLIdentifierPreparer

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mysql_dbapi_version = self._parse_dbapi_version(self.dbapi.__version__) if self.dbapi is not None and hasattr(self.dbapi, '__version__') else (0, 0, 0)

    def _parse_dbapi_version(self, version):
        m = re.match('(\\d+)\\.(\\d+)(?:\\.(\\d+))?', version)
        if m:
            return tuple((int(x) for x in m.group(1, 2, 3) if x is not None))
        else:
            return (0, 0, 0)

    @util.langhelpers.memoized_property
    def supports_server_side_cursors(self):
        try:
            cursors = __import__('MySQLdb.cursors').cursors
            self._sscursor = cursors.SSCursor
            return True
        except (ImportError, AttributeError):
            return False

    @classmethod
    def import_dbapi(cls):
        return __import__('MySQLdb')

    def on_connect(self):
        super_ = super().on_connect()

        def on_connect(conn):
            if super_ is not None:
                super_(conn)
            charset_name = conn.character_set_name()
            if charset_name is not None:
                cursor = conn.cursor()
                cursor.execute('SET NAMES %s' % charset_name)
                cursor.close()
        return on_connect

    def do_ping(self, dbapi_connection):
        dbapi_connection.ping()
        return True

    def do_executemany(self, cursor, statement, parameters, context=None):
        rowcount = cursor.executemany(statement, parameters)
        if context is not None:
            context._rowcount = rowcount

    def _check_unicode_returns(self, connection):
        collation = connection.exec_driver_sql("show collation where %s = 'utf8mb4' and %s = 'utf8mb4_bin'" % (self.identifier_preparer.quote('Charset'), self.identifier_preparer.quote('Collation'))).scalar()
        has_utf8mb4_bin = self.server_version_info > (5,) and collation
        if has_utf8mb4_bin:
            additional_tests = [sql.collate(sql.cast(sql.literal_column("'test collated returns'"), TEXT(charset='utf8mb4')), 'utf8mb4_bin')]
        else:
            additional_tests = []
        return super()._check_unicode_returns(connection, additional_tests)

    def create_connect_args(self, url, _translate_args=None):
        if _translate_args is None:
            _translate_args = dict(database='db', username='user', password='passwd')
        opts = url.translate_connect_args(**_translate_args)
        opts.update(url.query)
        util.coerce_kw_type(opts, 'compress', bool)
        util.coerce_kw_type(opts, 'connect_timeout', int)
        util.coerce_kw_type(opts, 'read_timeout', int)
        util.coerce_kw_type(opts, 'write_timeout', int)
        util.coerce_kw_type(opts, 'client_flag', int)
        util.coerce_kw_type(opts, 'local_infile', int)
        util.coerce_kw_type(opts, 'use_unicode', bool)
        util.coerce_kw_type(opts, 'charset', str)
        ssl = {}
        keys = [('ssl_ca', str), ('ssl_key', str), ('ssl_cert', str), ('ssl_capath', str), ('ssl_cipher', str), ('ssl_check_hostname', bool)]
        for key, kw_type in keys:
            if key in opts:
                ssl[key[4:]] = opts[key]
                util.coerce_kw_type(ssl, key[4:], kw_type)
                del opts[key]
        if ssl:
            opts['ssl'] = ssl
        client_flag = opts.get('client_flag', 0)
        client_flag_found_rows = self._found_rows_client_flag()
        if client_flag_found_rows is not None:
            client_flag |= client_flag_found_rows
            opts['client_flag'] = client_flag
        return [[], opts]

    def _found_rows_client_flag(self):
        if self.dbapi is not None:
            try:
                CLIENT_FLAGS = __import__(self.dbapi.__name__ + '.constants.CLIENT').constants.CLIENT
            except (AttributeError, ImportError):
                return None
            else:
                return CLIENT_FLAGS.FOUND_ROWS
        else:
            return None

    def _extract_error_code(self, exception):
        return exception.args[0]

    def _detect_charset(self, connection):
        """Sniff out the character set in use for connection results."""
        try:
            cset_name = connection.connection.character_set_name
        except AttributeError:
            util.warn("No 'character_set_name' can be detected with this MySQL-Python version; please upgrade to a recent version of MySQL-Python.  Assuming latin1.")
            return 'latin1'
        else:
            return cset_name()

    def get_isolation_level_values(self, dbapi_connection):
        return ('SERIALIZABLE', 'READ UNCOMMITTED', 'READ COMMITTED', 'REPEATABLE READ', 'AUTOCOMMIT')

    def set_isolation_level(self, dbapi_connection, level):
        if level == 'AUTOCOMMIT':
            dbapi_connection.autocommit(True)
        else:
            dbapi_connection.autocommit(False)
            super().set_isolation_level(dbapi_connection, level)