import codecs, re
from reportlab.lib.utils import isSeq, isBytes, isStr
from reportlab.lib import colors
from reportlab.lib.normalDate import NormalDate
class _isNumber(Validator):

    def test(self, x):
        if isinstance(x, (float, int)):
            return True
        return self.normalizeTest(x)

    def normalize(self, x):
        try:
            return float(x)
        except:
            return int(x)