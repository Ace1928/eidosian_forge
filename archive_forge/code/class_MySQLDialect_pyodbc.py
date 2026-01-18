import re
from .base import MySQLDialect
from .base import MySQLExecutionContext
from .types import TIME
from ... import exc
from ... import util
from ...connectors.pyodbc import PyODBCConnector
from ...sql.sqltypes import Time
class MySQLDialect_pyodbc(PyODBCConnector, MySQLDialect):
    supports_statement_cache = True
    colspecs = util.update_copy(MySQLDialect.colspecs, {Time: _pyodbcTIME})
    supports_unicode_statements = True
    execution_ctx_cls = MySQLExecutionContext_pyodbc
    pyodbc_driver_name = 'MySQL'

    def _detect_charset(self, connection):
        """Sniff out the character set in use for connection results."""
        self._connection_charset = None
        try:
            value = self._fetch_setting(connection, 'character_set_client')
            if value:
                return value
        except exc.DBAPIError:
            pass
        util.warn('Could not detect the connection character set.  Assuming latin1.')
        return 'latin1'

    def _get_server_version_info(self, connection):
        return MySQLDialect._get_server_version_info(self, connection)

    def _extract_error_code(self, exception):
        m = re.compile('\\((\\d+)\\)').search(str(exception.args))
        c = m.group(1)
        if c:
            return int(c)
        else:
            return None

    def on_connect(self):
        super_ = super().on_connect()

        def on_connect(conn):
            if super_ is not None:
                super_(conn)
            pyodbc_SQL_CHAR = 1
            pyodbc_SQL_WCHAR = -8
            conn.setdecoding(pyodbc_SQL_CHAR, encoding='utf-8')
            conn.setdecoding(pyodbc_SQL_WCHAR, encoding='utf-8')
            conn.setencoding(encoding='utf-8')
        return on_connect