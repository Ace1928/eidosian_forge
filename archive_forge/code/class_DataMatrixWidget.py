from reportlab.graphics.barcode.common import Barcode
from reportlab.lib.utils import asBytes
from reportlab.platypus.paraparser import _num as paraparser_num
from reportlab.graphics.widgetbase import Widget
from reportlab.lib.validators import isColor, isString, isColorOrNone, isNumber, isBoxAnchor
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.lib.colors import toColor
from reportlab.graphics.shapes import Group, Rect
class DataMatrixWidget(Widget, _DMTXCheck):
    codeName = 'DataMatrix'
    _attrMap = AttrMap(BASE=Widget, value=AttrMapValue(isString, desc='Datamatrix data'), x=AttrMapValue(isNumber, desc='x-coord'), y=AttrMapValue(isNumber, desc='y-coord'), color=AttrMapValue(isColor, desc='foreground color'), bgColor=AttrMapValue(isColorOrNone, desc='background color'), encoding=AttrMapValue(isString, desc='encoding'), size=AttrMapValue(isString, desc='size'), cellSize=AttrMapValue(isString, desc='cellSize'), anchor=AttrMapValue(isBoxAnchor, desc='anchor pooint for x,y'))
    _defaults = dict(x=('0', _numConv), y=('0', _numConv), color=('black', toColor), bgColor=(None, lambda _: toColor(_) if _ is not None else _), encoding=('Ascii', None), size=('SquareAuto', None), cellSize=('5x5', None), anchor=('sw', None))

    def __init__(self, value='Hello Cruel World!', **kwds):
        self.pylibdmtx_check()
        self.value = value
        for k, (d, c) in self._defaults.items():
            v = kwds.pop(k, d)
            if c:
                v = c(v)
            setattr(self, k, v)

    def rect(self, x, y, w, h, fill=1, stroke=0):
        self._gadd(Rect(x, y, w, h, strokeColor=None, fillColor=self._fillColor))

    def saveState(self, *args, **kwds):
        pass
    restoreState = setStrokeColor = saveState

    def setFillColor(self, c):
        self._fillColor = c

    def draw(self):
        m = DataMatrix(value=self.value, **{k: getattr(self, k) for k in self._defaults})
        m.canv = self
        m.y += m.height
        g = Group()
        self._gadd = g.add
        m.draw()
        return g