import codecs, re
from reportlab.lib.utils import isSeq, isBytes, isStr
from reportlab.lib import colors
from reportlab.lib.normalDate import NormalDate
class _isInt(Validator):

    def test(self, x):
        if not isinstance(x, int) and (not isStr(x)):
            return False
        return self.normalizeTest(x)

    def normalize(self, x):
        return int(x.decode('utf8') if isBytes(x) else x)