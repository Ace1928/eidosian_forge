import sys
import traceback
from mako import compat
from mako import util
class NameConflictError(MakoException):
    """raised when a reserved word is used inappropriately"""