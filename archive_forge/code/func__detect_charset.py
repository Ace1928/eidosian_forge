from .base import BIT
from .base import MySQLDialect
from .mysqldb import MySQLDialect_mysqldb
from ... import util
def _detect_charset(self, connection):
    return connection.connection.charset