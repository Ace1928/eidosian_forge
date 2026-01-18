from .mysqldb import MySQLDialect_mysqldb
from ...util import langhelpers
class MySQLDialect_pymysql(MySQLDialect_mysqldb):
    driver = 'pymysql'
    supports_statement_cache = True
    description_encoding = None

    @langhelpers.memoized_property
    def supports_server_side_cursors(self):
        try:
            cursors = __import__('pymysql.cursors').cursors
            self._sscursor = cursors.SSCursor
            return True
        except (ImportError, AttributeError):
            return False

    @classmethod
    def import_dbapi(cls):
        return __import__('pymysql')

    @langhelpers.memoized_property
    def _send_false_to_ping(self):
        """determine if pymysql has deprecated, changed the default of,
        or removed the 'reconnect' argument of connection.ping().

        See #10492 and
        https://github.com/PyMySQL/mysqlclient/discussions/651#discussioncomment-7308971
        for background.

        """
        try:
            Connection = __import__('pymysql.connections').connections.Connection
        except (ImportError, AttributeError):
            return True
        else:
            insp = langhelpers.get_callable_argspec(Connection.ping)
            try:
                reconnect_arg = insp.args[1]
            except IndexError:
                return False
            else:
                return reconnect_arg == 'reconnect' and (not insp.defaults or insp.defaults[0] is not False)

    def do_ping(self, dbapi_connection):
        if self._send_false_to_ping:
            dbapi_connection.ping(False)
        else:
            dbapi_connection.ping()
        return True

    def create_connect_args(self, url, _translate_args=None):
        if _translate_args is None:
            _translate_args = dict(username='user')
        return super().create_connect_args(url, _translate_args=_translate_args)

    def is_disconnect(self, e, connection, cursor):
        if super().is_disconnect(e, connection, cursor):
            return True
        elif isinstance(e, self.dbapi.Error):
            str_e = str(e).lower()
            return 'already closed' in str_e or 'connection was killed' in str_e
        else:
            return False

    def _extract_error_code(self, exception):
        if isinstance(exception.args[0], Exception):
            exception = exception.args[0]
        return exception.args[0]