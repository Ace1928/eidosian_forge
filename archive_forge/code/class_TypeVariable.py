import gast as ast
from copy import deepcopy
from numpy import floating, integer, complexfloating
from pythran.tables import MODULES, attributes
import pythran.typing as typing
from pythran.syntax import PythranSyntaxError
from pythran.utils import isnum
class TypeVariable(object):
    """A type variable standing for an arbitrary type.

    All type variables have a unique id, but names are only assigned lazily,
    when required.
    """
    _cached_names = {}

    def __init__(self):
        self.instance = None
        self.name = None

    def __str__(self):
        if self.instance:
            return str(self.instance)
        else:
            return 'T{}'.format(TypeVariable._cached_names.setdefault(self, len(TypeVariable._cached_names)))