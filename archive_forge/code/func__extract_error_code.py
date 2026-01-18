from .base import BIT
from .base import MySQLDialect
from .mysqldb import MySQLDialect_mysqldb
from ... import util
def _extract_error_code(self, exception):
    return exception.errno