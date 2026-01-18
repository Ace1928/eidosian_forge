from .mysqldb import MySQLDialect_mysqldb
from ...util import langhelpers
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