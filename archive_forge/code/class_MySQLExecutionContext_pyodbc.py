import re
from .base import MySQLDialect
from .base import MySQLExecutionContext
from .types import TIME
from ... import exc
from ... import util
from ...connectors.pyodbc import PyODBCConnector
from ...sql.sqltypes import Time
class MySQLExecutionContext_pyodbc(MySQLExecutionContext):

    def get_lastrowid(self):
        cursor = self.create_cursor()
        cursor.execute('SELECT LAST_INSERT_ID()')
        lastrowid = cursor.fetchone()[0]
        cursor.close()
        return lastrowid