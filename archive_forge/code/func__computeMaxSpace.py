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
def _computeMaxSpace(self, size, required):
    """helper for madmen who want to put stuff inside their barcharts
        basically after _computebarPositions we slide a line of length size
        down the bar profile on either side of the bars to find the
        maximum space. If the space at any point is >= required then we're
        done. Otherwise we return the largest space location and amount.
        """
    flipXY = self._flipXY
    self._computeBarPositions()
    lenData = len(self.data)
    BP = self._barPositions
    C = []
    aC = C.append
    if flipXY:
        lo = self.x
        hi = lo + self.width
        end = self.y + self.height
        for bp in BP:
            for x, y, w, h in bp:
                v = x + w
                z = y + h
                aC((min(y, z), max(y, z), min(x, v) - lo, hi - max(x, v)))
    else:
        lo = self.y
        hi = lo + self.height
        end = self.x + self.width
        for bp in BP:
            for x, y, w, h in bp:
                v = y + h
                z = x + w
                aC((min(x, z), max(x, z), min(y, v) - lo, hi - max(y, v)))
    C.sort()
    R = [C[0]]
    for c in C:
        r = R[-1]
        if r[0] < c[1] and c[0] < r[1]:
            R[-1] = (min(r[0], c[0]), max(r[1], c[1]), min(r[2], c[2]), min(r[3], c[3]))
        else:
            R.append(c)
    C = R
    maxS = -2147483647
    maxP = None
    nC = len(C)
    for i, ci in enumerate(C):
        v0 = ci[0]
        v1 = v0 + size
        if v1 > end:
            break
        j = i
        alo = ahi = 2147483647
        while j < nC and C[j][1] <= v1:
            alo = min(C[j][2], alo)
            ahi = min(C[j][3], ahi)
            j += 1
        if alo > ahi:
            if alo > maxS:
                maxS = alo
                maxP = flipXY and (lo, v0, lo + alo, v0 + size, 0) or (v0, lo, v0 + size, lo + alo, 0)
                if maxS >= required:
                    break
        elif ahi > maxS:
            maxS = ahi
            maxP = flipXY and (hi - ahi, v0, hi, v0 + size, 1) or (v0, hi - ahi, v0 + size, hi, 1)
            if maxS >= required:
                break
    return (maxS, maxP)