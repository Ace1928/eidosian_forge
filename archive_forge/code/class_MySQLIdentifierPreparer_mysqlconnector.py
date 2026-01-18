import re
from .base import BIT
from .base import MySQLCompiler
from .base import MySQLDialect
from .base import MySQLIdentifierPreparer
from ... import util
class MySQLIdentifierPreparer_mysqlconnector(MySQLIdentifierPreparer):

    @property
    def _double_percents(self):
        return False

    @_double_percents.setter
    def _double_percents(self, value):
        pass

    def _escape_identifier(self, value):
        value = value.replace(self.escape_quote, self.escape_to_quote)
        return value