import copy, functools
from ast import literal_eval
from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isNumberOrNone, isColorOrNone, isString,\
from reportlab.lib.utils import isStr, yieldNoneSplits
from reportlab.graphics.widgets.markers import uSymbol2Symbol, isSymbol
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.graphics.widgetbase import TypedPropertyCollection, PropHolder, tpcGetItem
from reportlab.graphics.shapes import Line, Rect, Group, Drawing, PolyLine
from reportlab.graphics.charts.axes import XCategoryAxis, YValueAxis, YCategoryAxis, XValueAxis
from reportlab.graphics.charts.textlabels import BarChartLabel, NoneOrInstanceOfNA_Label
from reportlab.graphics.charts.areas import PlotArea
from reportlab.graphics.charts.legends import _objStr
from reportlab import cmp
def _drawFinish(self):
    """finalize the drawing of a barchart"""
    cA = self.categoryAxis
    vA = self.valueAxis
    cA.configure(self._configureData)
    self.calcBarPositions()
    g = Group()
    zIndex = getattr(self, 'zIndexOverrides', None)
    if not zIndex:
        g.add(self.makeBackground())
        cAdgl = getattr(cA, 'drawGridLast', False)
        vAdgl = getattr(vA, 'drawGridLast', False)
        if not cAdgl:
            cA.makeGrid(g, parent=self, dim=vA.getGridDims)
        if not vAdgl:
            vA.makeGrid(g, parent=self, dim=cA.getGridDims)
        g.add(self.makeBars())
        g.add(cA)
        g.add(vA)
        if cAdgl:
            cA.makeGrid(g, parent=self, dim=vA.getGridDims)
        if vAdgl:
            vA.makeGrid(g, parent=self, dim=cA.getGridDims)
        for a in getattr(self, 'annotations', ()):
            g.add(a(self, cA.scale, vA.scale))
    else:
        Z = dict(background=0, categoryAxis=1, valueAxis=2, bars=3, barLabels=4, categoryAxisGrid=5, valueAxisGrid=6, annotations=7)
        for z in zIndex.strip().split(','):
            z = z.strip()
            if not z:
                continue
            try:
                k, v = z.split('=')
            except:
                raise ValueError('Badly formatted zIndex clause %r in %r\nallowed variables are\n%s' % (z, zIndex, '\n'.join(['%s=%r' % (k, Z[k]) for k in sorted(Z.keys())])))
            if k not in Z:
                raise ValueError('Unknown zIndex variable %r in %r\nallowed variables are\n%s' % (k, Z, '\n'.join(['%s=%r' % (k, Z[k]) for k in sorted(Z.keys())])))
            try:
                v = literal_eval(v)
                assert isinstance(v, (float, int))
            except:
                raise ValueError('Bad zIndex value %r in clause %r of zIndex\nallowed variables are\n%s' % (v, z, zIndex, '\n'.join(['%s=%r' % (k, Z[k]) for k in sorted(Z.keys())])))
            Z[k] = v
        Z = [(v, k) for k, v in Z.items()]
        Z.sort()
        b = self.makeBars()
        bl = b.contents.pop(-1)
        for v, k in Z:
            if k == 'background':
                g.add(self.makeBackground())
            elif k == 'categoryAxis':
                g.add(cA)
            elif k == 'categoryAxisGrid':
                cA.makeGrid(g, parent=self, dim=vA.getGridDims)
            elif k == 'valueAxis':
                g.add(vA)
            elif k == 'valueAxisGrid':
                vA.makeGrid(g, parent=self, dim=cA.getGridDims)
            elif k == 'bars':
                g.add(b)
            elif k == 'barLabels':
                g.add(bl)
            elif k == 'annotations':
                for a in getattr(self, 'annotations', ()):
                    g.add(a(self, cA.scale, vA.scale))
    del self._configureData
    return g