import gast as ast
from copy import deepcopy
from numpy import floating, integer, complexfloating
from pythran.tables import MODULES, attributes
import pythran.typing as typing
from pythran.syntax import PythranSyntaxError
from pythran.utils import isnum
class MultiType(object):
    """A binary type constructor which builds function types"""

    def __init__(self, types):
        self.name = 'multitype'
        self.types = types

    def __str__(self):
        return '\n'.join(sorted(map(str, self.types)))