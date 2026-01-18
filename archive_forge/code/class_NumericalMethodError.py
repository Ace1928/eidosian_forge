from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
class NumericalMethodError(Exception):

    def __init__(self, method):
        self.method = method

    def __str__(self):
        return 'Method %s only supported for numerical values' % self.method