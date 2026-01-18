import re
from .base import BIT
from .base import MySQLCompiler
from .base import MySQLDialect
from .base import MySQLIdentifierPreparer
from ... import util
class _myconnpyBIT(BIT):

    def result_processor(self, dialect, coltype):
        """MySQL-connector already converts mysql bits, so."""
        return None