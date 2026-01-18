from reportlab.lib.validators import isInt, isNumber, isString, isColorOrNone, isBoolean, EitherOr, isNumberOrNone
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.lib.colors import black
from reportlab.lib.utils import rl_exec
from reportlab.graphics.shapes import Rect, Group, String
from reportlab.graphics.charts.areas import PlotArea
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.barcode.widgets import BarcodeStandard93
class _BarcodeWidget(PlotArea):
    _attrMap = AttrMap(BASE=PlotArea, barStrokeColor=AttrMapValue(isColorOrNone, desc='Color of bar borders.'), barFillColor=AttrMapValue(isColorOrNone, desc='Color of bar interior areas.'), barStrokeWidth=AttrMapValue(isNumber, desc='Width of bar borders.'), value=AttrMapValue(EitherOr((isString, isNumber)), desc='Value.'), textColor=AttrMapValue(isColorOrNone, desc='Color of human readable text.'), valid=AttrMapValue(isBoolean), validated=AttrMapValue(isString, desc='validated form of input'), encoded=AttrMapValue(None, desc='encoded form of input'), decomposed=AttrMapValue(isString, desc='decomposed form of input'), canv=AttrMapValue(None, desc='temporarily used for internal methods'), gap=AttrMapValue(isNumberOrNone, desc='Width of inter character gaps.'))
    textColor = barFillColor = black
    barStrokeColor = None
    barStrokeWidth = 0
    _BCC = None

    def __init__(self, _value='', **kw):
        PlotArea.__init__(self)
        if 'width' in self.__dict__:
            del self.__dict__['width']
        if 'height' in self.__dict__:
            del self.__dict__['height']
        self.x = self.y = 0
        kw.setdefault('value', _value)
        self._BCC.__init__(self, **kw)

    def rect(self, x, y, w, h, **kw):
        for k, v in (('strokeColor', self.barStrokeColor), ('strokeWidth', self.barStrokeWidth), ('fillColor', self.barFillColor)):
            kw.setdefault(k, v)
        self._Gadd(Rect(self.x + x, self.y + y, w, h, **kw))

    def draw(self):
        if not self._BCC:
            raise NotImplementedError('Abstract class %s cannot be drawn' % self.__class__.__name__)
        self.canv = self
        G = Group()
        self._Gadd = G.add
        self._Gadd(Rect(self.x, self.y, self.width, self.height, fillColor=None, strokeColor=None, strokeWidth=0.0001))
        self._BCC.draw(self)
        del self.canv, self._Gadd
        return G

    def annotate(self, x, y, text, fontName, fontSize, anchor='middle'):
        self._Gadd(String(self.x + x, self.y + y, text, fontName=fontName, fontSize=fontSize, textAnchor=anchor, fillColor=self.textColor))