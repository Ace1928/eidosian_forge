from reportlab.graphics.shapes import Group, String, Rect
from reportlab.lib import colors
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.validators import isNumber, isColor, isString, Validator, isBoolean, NoneOr
from reportlab.lib.attrmap import *
from reportlab.graphics.charts.areas import PlotArea
from reportlab.lib.units import mm
from reportlab.lib.utils import asNative
class _ndigits(Validator):

    def test(self, x):
        return type(x) is str and len(x) <= n and (len([c for c in x if c in '0123456789']) == n)