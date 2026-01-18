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
def _rawDraw(self):
    _text = self._text
    self._text = _text or ''
    self.computeSize()
    self._text = _text
    g = Group()
    g.translate(self.x + self.dx, self.y + self.dy)
    g.rotate(self.angle)
    x = self._left
    if self.boxFillColor or (self.boxStrokeColor and self.boxStrokeWidth):
        g.add(Rect(self._left - self.leftPadding, self._bottom - self.bottomPadding, self._width, self._height, strokeColor=self.boxStrokeColor, strokeWidth=self.boxStrokeWidth, fillColor=self.boxFillColor))
    g1 = Group()
    g1.translate(x, self._top - self._eheight)
    g1.add(self._ddf(self._obj))
    g.add(g1)
    return g