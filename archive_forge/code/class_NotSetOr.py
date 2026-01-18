import codecs, re
from reportlab.lib.utils import isSeq, isBytes, isStr
from reportlab.lib import colors
from reportlab.lib.normalDate import NormalDate
class NotSetOr(EitherOr):
    _not_set = object()

    def test(self, x):
        return x is NotSetOr._not_set or super().test(x)

    @staticmethod
    def conditionalValue(v, a):
        return a if v is NotSetOr._not_set else v