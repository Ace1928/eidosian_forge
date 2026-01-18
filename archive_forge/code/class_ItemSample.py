import math
from .. import functions as fn
from ..icons import invisibleEye
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .BarGraphItem import BarGraphItem
from .GraphicsWidget import GraphicsWidget
from .GraphicsWidgetAnchor import GraphicsWidgetAnchor
from .LabelItem import LabelItem
from .PlotDataItem import PlotDataItem
from .ScatterPlotItem import ScatterPlotItem, drawSymbol
class ItemSample(GraphicsWidget):
    """Class responsible for drawing a single item in a LegendItem (sans label)
    """

    def __init__(self, item):
        GraphicsWidget.__init__(self)
        self.item = item

    def boundingRect(self):
        return QtCore.QRectF(0, 0, 20, 20)

    def paint(self, p, *args):
        opts = self.item.opts
        if opts.get('antialias'):
            p.setRenderHint(p.RenderHint.Antialiasing)
        visible = self.item.isVisible()
        if not visible:
            icon = invisibleEye.qicon
            p.drawPixmap(QtCore.QPoint(1, 1), icon.pixmap(18, 18))
            return
        if not isinstance(self.item, ScatterPlotItem):
            p.setPen(fn.mkPen(opts['pen']))
            p.drawLine(0, 11, 20, 11)
            if opts.get('fillLevel', None) is not None and opts.get('fillBrush', None) is not None:
                p.setBrush(fn.mkBrush(opts['fillBrush']))
                p.setPen(fn.mkPen(opts['pen']))
                p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(2, 18), QtCore.QPointF(18, 2), QtCore.QPointF(18, 18)]))
        symbol = opts.get('symbol', None)
        if symbol is not None:
            if isinstance(self.item, PlotDataItem):
                opts = self.item.scatter.opts
            p.translate(10, 10)
            drawSymbol(p, symbol, opts['size'], fn.mkPen(opts['pen']), fn.mkBrush(opts['brush']))
        if isinstance(self.item, BarGraphItem):
            p.setBrush(fn.mkBrush(opts['brush']))
            p.drawRect(QtCore.QRectF(2, 2, 18, 18))

    def mouseClickEvent(self, event):
        """Use the mouseClick event to toggle the visibility of the plotItem
        """
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            visible = self.item.isVisible()
            self.item.setVisible(not visible)
        event.accept()
        self.update()