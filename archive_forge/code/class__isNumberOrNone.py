import codecs, re
from reportlab.lib.utils import isSeq, isBytes, isStr
from reportlab.lib import colors
from reportlab.lib.normalDate import NormalDate
class _isNumberOrNone(_isNumber):

    def test(self, x):
        return x is None or isNumber(x)

    def normalize(self, x):
        if x is None:
            return x
        return _isNumber.normalize(x)