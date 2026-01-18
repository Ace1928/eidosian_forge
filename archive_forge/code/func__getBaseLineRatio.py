from reportlab.lib import colors
from reportlab.lib.utils import simpleSplit
from reportlab.lib.validators import isNumber, isNumberOrNone, OneOf, isColorOrNone, isString, \
from reportlab.lib.attrmap import *
from reportlab.pdfbase.pdfmetrics import stringWidth, getAscentDescent
from reportlab.graphics.shapes import Drawing, Group, Circle, Rect, String, STATE_DEFAULTS
from reportlab.graphics.widgetbase import Widget, PropHolder
from reportlab.graphics.shapes import DirectDraw
from reportlab.platypus import XPreformatted, Flowable
from reportlab.lib.styles import ParagraphStyle, PropertySet
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER
from ..utils import text2Path as _text2Path   #here for continuity
from reportlab.graphics.charts.utils import CustomDrawChanger
def _getBaseLineRatio(self):
    if self.useAscentDescent:
        self._ascent, self._descent = getAscentDescent(self.fontName, self.fontSize)
        self._baselineRatio = self._ascent / (self._ascent - self._descent)
    else:
        self._baselineRatio = 1 / 1.2