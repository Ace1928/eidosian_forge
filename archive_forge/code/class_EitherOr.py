import codecs, re
from reportlab.lib.utils import isSeq, isBytes, isStr
from reportlab.lib import colors
from reportlab.lib.normalDate import NormalDate
class EitherOr(Validator):

    def __init__(self, tests, name=None):
        if not isSeq(tests):
            tests = (tests,)
        self._tests = tests
        if name:
            self._str = name

    def test(self, x):
        for t in self._tests:
            if t(x):
                return True
        return False