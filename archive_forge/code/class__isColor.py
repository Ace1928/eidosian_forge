import codecs, re
from reportlab.lib.utils import isSeq, isBytes, isStr
from reportlab.lib import colors
from reportlab.lib.normalDate import NormalDate
class _isColor(Validator):
    """Color validator class."""

    def test(self, x):
        return isinstance(x, colors.Color)