from reportlab.graphics.shapes import Group, String, Rect
from reportlab.lib import colors
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.validators import isNumber, isColor, isString, Validator, isBoolean, NoneOr
from reportlab.lib.attrmap import *
from reportlab.graphics.charts.areas import PlotArea
from reportlab.lib.units import mm
from reportlab.lib.utils import asNative
def _encode_left(self, s, a):
    check = self._checkdigit(s)
    cp = self._lhconvert[check]
    _left = self._left
    _sep = self._sep
    z = ord('0')
    full_code = []
    for i, c in enumerate(s):
        full_code.append(_left[cp[i]][ord(c) - z])
    a(_sep.join(full_code))