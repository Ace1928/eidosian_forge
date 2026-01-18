import codecs, re
from reportlab.lib.utils import isSeq, isBytes, isStr
from reportlab.lib import colors
from reportlab.lib.normalDate import NormalDate
class isSubclassOf(Validator):

    def __init__(self, klass=None):
        self._klass = klass

    def test(self, x):
        return isinstance(x, type) and issubclass(x, self._klass)