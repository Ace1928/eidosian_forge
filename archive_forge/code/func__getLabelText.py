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
def _getLabelText(self, rowNo, colNo):
    """return formatted label text"""
    labelFmt = self.barLabelFormat
    if isinstance(labelFmt, (list, tuple)):
        labelFmt = labelFmt[rowNo]
        if isinstance(labelFmt, (list, tuple)):
            labelFmt = labelFmt[colNo]
    if labelFmt is None:
        labelText = None
    elif labelFmt == 'values':
        labelText = self.barLabelArray[rowNo][colNo]
    elif isStr(labelFmt):
        labelText = labelFmt % self.data[rowNo][colNo]
    elif hasattr(labelFmt, '__call__'):
        labelText = labelFmt(self.data[rowNo][colNo])
    else:
        msg = 'Unknown formatter type %s, expected string or function' % labelFmt
        raise Exception(msg)
    return labelText