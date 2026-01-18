import codecs, re
from reportlab.lib.utils import isSeq, isBytes, isStr
from reportlab.lib import colors
from reportlab.lib.normalDate import NormalDate
def _test_patterns(self, x):
    v = x in self._enum
    if v:
        return True
    for p in self._patterns:
        v = p.match(x)
        if v:
            return True
    return False