import codecs, re
from reportlab.lib.utils import isSeq, isBytes, isStr
from reportlab.lib import colors
from reportlab.lib.normalDate import NormalDate
class NoneOr(EitherOr):

    def test(self, x):
        return x is None or super().test(x)