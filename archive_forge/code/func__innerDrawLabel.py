from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.lib.utils import flatten, isStr
from reportlab.graphics.shapes import Drawing, Group, Rect, PolyLine, Polygon, _SetKeyWordArgs
from reportlab.graphics.widgetbase import TypedPropertyCollection, PropHolder, tpcGetItem
from reportlab.graphics.charts.textlabels import Label
from reportlab.graphics.charts.axes import XValueAxis, YValueAxis, AdjYValueAxis, NormalDateXValueAxis
from reportlab.graphics.charts.utils import *
from reportlab.graphics.widgets.markers import uSymbol2Symbol, makeMarker
from reportlab.graphics.widgets.grids import Grid, DoubleGrid, ShadedPolygon
from reportlab.pdfbase.pdfmetrics import stringWidth, getFont
from reportlab.graphics.charts.areas import PlotArea
from .utils import FillPairedData
from reportlab.graphics.charts.linecharts import AbstractLineChart
def _innerDrawLabel(self, rowNo, colNo, x, y):
    """Draw a label for a given item in the list."""
    labelFmt = self.lineLabelFormat
    labelValue = self.data[rowNo][colNo][1]
    if labelFmt is None:
        labelText = None
    elif isinstance(labelFmt, str):
        if labelFmt == 'values':
            labelText = self.lineLabelArray[rowNo][colNo]
        else:
            labelText = labelFmt % labelValue
    elif hasattr(labelFmt, '__call__'):
        if not hasattr(labelFmt, '__labelFmtEX__'):
            labelText = labelFmt(labelValue)
        else:
            labelText = labelFmt(self, rowNo, colNo, x, y)
    else:
        raise ValueError('Unknown formatter type %s, expected string or function' % labelFmt)
    if labelText:
        label = self.lineLabels[rowNo, colNo]
        if not label.visible:
            return
        if y > 0:
            label.setOrigin(x, y + self.lineLabelNudge)
        else:
            label.setOrigin(x, y - self.lineLabelNudge)
        label.setText(labelText)
    else:
        label = None
    return label