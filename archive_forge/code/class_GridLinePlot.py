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
class GridLinePlot(SimpleTimeSeriesPlot):
    """A customized version of SimpleTimeSeriesSPlot.
    It uses NormalDateXValueAxis() and AdjYValueAxis() for the X and Y axes.
    The chart has a default grid background with thin horizontal lines
    aligned with the tickmarks (and labels). You can change the back-
    ground to be any Grid or ShadedRect, or scale the whole chart.
    If you do provide a background, you can specify the colours of the
    stripes with 'background.stripeColors'.
    """
    _attrMap = AttrMap(BASE=LinePlot, background=AttrMapValue(None, desc='Background for chart area (now Grid or ShadedRect).'), scaleFactor=AttrMapValue(isNumberOrNone, desc='Scalefactor to apply to whole drawing.'))

    def __init__(self):
        from reportlab.lib import colors
        SimpleTimeSeriesPlot.__init__(self)
        self.scaleFactor = None
        self.background = Grid()
        self.background.orientation = 'horizontal'
        self.background.useRects = 0
        self.background.useLines = 1
        self.background.strokeWidth = 0.5
        self.background.strokeColor = colors.black

    def demo(self, drawing=None):
        from reportlab.lib import colors
        if not drawing:
            drawing = Drawing(400, 200)
        lp = GridLinePlot()
        lp.x = 50
        lp.y = 50
        lp.height = 125
        lp.width = 300
        lp.data = _monthlyIndexData
        lp.joinedLines = 1
        lp.strokeColor = colors.black
        c0 = colors.PCMYKColor(100, 65, 0, 30, spotName='PANTONE 288 CV', density=100)
        lp.lines[0].strokeColor = c0
        lp.lines[0].strokeWidth = 2
        lp.lines[0].strokeDashArray = None
        c1 = colors.PCMYKColor(0, 79, 91, 0, spotName='PANTONE Wm Red CV', density=100)
        lp.lines[1].strokeColor = c1
        lp.lines[1].strokeWidth = 1
        lp.lines[1].strokeDashArray = [3, 1]
        lp.xValueAxis.labels.fontSize = 10
        lp.xValueAxis.labels.textAnchor = 'start'
        lp.xValueAxis.labels.boxAnchor = 'w'
        lp.xValueAxis.labels.angle = -45
        lp.xValueAxis.labels.dx = 0
        lp.xValueAxis.labels.dy = -8
        lp.xValueAxis.xLabelFormat = '{mm}/{yy}'
        lp.yValueAxis.labelTextFormat = '%5d%% '
        lp.yValueAxis.tickLeft = 5
        lp.yValueAxis.labels.fontSize = 10
        lp.background = Grid()
        lp.background.stripeColors = [colors.pink, colors.lightblue]
        lp.background.orientation = 'vertical'
        drawing.add(lp, 'plot')
        return drawing

    def draw(self):
        xva, yva = (self.xValueAxis, self.yValueAxis)
        if xva:
            xva.joinAxis = yva
        if yva:
            yva.joinAxis = xva
        yva.setPosition(self.x, self.y, self.height)
        yva.configure(self.data)
        xAxisCrossesAt = yva.scale(0)
        if xAxisCrossesAt > self.y + self.height or xAxisCrossesAt < self.y:
            y = self.y
        else:
            y = xAxisCrossesAt
        xva.setPosition(self.x, y, self.width)
        xva.configure(self.data)
        back = self.background
        if isinstance(back, Grid):
            if back.orientation == 'vertical' and xva._tickValues:
                xpos = list(map(xva.scale, [xva._valueMin] + xva._tickValues))
                steps = []
                for i in range(len(xpos) - 1):
                    steps.append(xpos[i + 1] - xpos[i])
                back.deltaSteps = steps
            elif back.orientation == 'horizontal' and yva._tickValues:
                ypos = list(map(yva.scale, [yva._valueMin] + yva._tickValues))
                steps = []
                for i in range(len(ypos) - 1):
                    steps.append(ypos[i + 1] - ypos[i])
                back.deltaSteps = steps
        elif isinstance(back, DoubleGrid):
            back.grid0.x = self.x
            back.grid0.y = self.y
            back.grid0.width = self.width
            back.grid0.height = self.height
            back.grid1.x = self.x
            back.grid1.y = self.y
            back.grid1.width = self.width
            back.grid1.height = self.height
            if back.grid0.orientation == 'vertical' and xva._tickValues:
                xpos = list(map(xva.scale, [xva._valueMin] + xva._tickValues))
                steps = []
                for i in range(len(xpos) - 1):
                    steps.append(xpos[i + 1] - xpos[i])
                back.grid0.deltaSteps = steps
            elif back.grid0.orientation == 'horizontal' and yva._tickValues:
                ypos = list(map(yva.scale, [yva._valueMin] + yva._tickValues))
                steps = []
                for i in range(len(ypos) - 1):
                    steps.append(ypos[i + 1] - ypos[i])
                back.grid0.deltaSteps = steps
            if back.grid1.orientation == 'vertical' and xva._tickValues:
                xpos = list(map(xva.scale, [xva._valueMin] + xva._tickValues))
                steps = []
                for i in range(len(xpos) - 1):
                    steps.append(xpos[i + 1] - xpos[i])
                back.grid1.deltaSteps = steps
            elif back.grid1.orientation == 'horizontal' and yva._tickValues:
                ypos = list(map(yva.scale, [yva._valueMin] + yva._tickValues))
                steps = []
                for i in range(len(ypos) - 1):
                    steps.append(ypos[i + 1] - ypos[i])
                back.grid1.deltaSteps = steps
        self.calcPositions()
        width, height, scaleFactor = (self.width, self.height, self.scaleFactor)
        if scaleFactor and scaleFactor != 1:
            g.transform = (scaleFactor, 0, 0, scaleFactor, 0, 0)
        else:
            g = Group()
        g.add(self.makeBackground())
        g.add(self.xValueAxis)
        g.add(self.yValueAxis)
        g.add(self.makeLines())
        return g