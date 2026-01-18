from reportlab.graphics.shapes import Group, String, Rect
from reportlab.lib import colors
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.validators import isNumber, isColor, isString, Validator, isBoolean, NoneOr
from reportlab.lib.attrmap import *
from reportlab.graphics.charts.areas import PlotArea
from reportlab.lib.units import mm
from reportlab.lib.utils import asNative
def _short_bar(self, i):
    i += 9 - self._lquiet
    return self.humanReadable and (12 < i < 41 or 43 < i < 73)