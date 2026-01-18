from .base import BIT
from .base import MySQLDialect
from .mysqldb import MySQLDialect_mysqldb
from ... import util
class MySQLDialect_cymysql(MySQLDialect_mysqldb):
    driver = 'cymysql'
    supports_statement_cache = True
    description_encoding = None
    supports_sane_rowcount = True
    supports_sane_multi_rowcount = False
    supports_unicode_statements = True
    colspecs = util.update_copy(MySQLDialect.colspecs, {BIT: _cymysqlBIT})

    @classmethod
    def import_dbapi(cls):
        return __import__('cymysql')

    def _detect_charset(self, connection):
        return connection.connection.charset

    def _extract_error_code(self, exception):
        return exception.errno

    def is_disconnect(self, e, connection, cursor):
        if isinstance(e, self.dbapi.OperationalError):
            return self._extract_error_code(e) in (2006, 2013, 2014, 2045, 2055)
        elif isinstance(e, self.dbapi.InterfaceError):
            return True
        else:
            return False