from reportlab.graphics.shapes import Group, String, Rect
from reportlab.lib import colors
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.validators import isNumber, isColor, isString, Validator, isBoolean, NoneOr
from reportlab.lib.attrmap import *
from reportlab.graphics.charts.areas import PlotArea
from reportlab.lib.units import mm
from reportlab.lib.utils import asNative
def _checkdigit(cls, num):
    z = ord('0')
    iSum = cls._0csw * sum([ord(x) - z for x in num[::2]]) + cls._1csw * sum([ord(x) - z for x in num[1::2]])
    return chr(z + iSum % 10)