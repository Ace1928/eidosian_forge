from decimal import Decimal
from MySQLdb._mysql import string_literal
from MySQLdb.constants import FIELD_TYPE, FLAG
from MySQLdb.times import (
from MySQLdb._exceptions import ProgrammingError
import array
def Thing2Literal(o, d):
    """Convert something into a SQL string literal.  If using
    MySQL-3.23 or newer, string_literal() is a method of the
    _mysql.MYSQL object, and this function will be overridden with
    that method when the connection is created."""
    return string_literal(o)