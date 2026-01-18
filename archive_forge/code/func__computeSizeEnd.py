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
def _computeSizeEnd(self, objH):
    self._height = self.height or objH + self.topPadding + self.bottomPadding
    self._ewidth = self._width - self.leftPadding - self.rightPadding
    self._eheight = self._height - self.topPadding - self.bottomPadding
    boxAnchor = self._getBoxAnchor()
    if boxAnchor in ['n', 'ne', 'nw']:
        self._top = -self.topPadding
    elif boxAnchor in ['s', 'sw', 'se']:
        self._top = self._height - self.topPadding
    else:
        self._top = 0.5 * self._eheight
    self._bottom = self._top - self._eheight
    if boxAnchor in ['ne', 'e', 'se']:
        self._left = self.leftPadding - self._width
    elif boxAnchor in ['nw', 'w', 'sw']:
        self._left = self.leftPadding
    else:
        self._left = -self._ewidth * 0.5
    self._right = self._left + self._ewidth