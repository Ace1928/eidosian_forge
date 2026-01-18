import weakref
from .. import functions as fn
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
class ConnectionItem(GraphicsObject):

    def __init__(self, source, target=None):
        GraphicsObject.__init__(self)
        self.setFlags(self.GraphicsItemFlag.ItemIsSelectable | self.GraphicsItemFlag.ItemIsFocusable)
        self.source = source
        self.target = target
        self.length = 0
        self.hovered = False
        self.path = None
        self.shapePath = None
        self.style = {'shape': 'line', 'color': (100, 100, 250), 'width': 1.0, 'hoverColor': (150, 150, 250), 'hoverWidth': 1.0, 'selectedColor': (200, 200, 0), 'selectedWidth': 3.0}
        self.source.getViewBox().addItem(self)
        self.updateLine()
        self.setZValue(0)

    def close(self):
        if self.scene() is not None:
            self.scene().removeItem(self)

    def setTarget(self, target):
        self.target = target
        self.updateLine()

    def setStyle(self, **kwds):
        self.style.update(kwds)
        if 'shape' in kwds:
            self.updateLine()
        else:
            self.update()

    def updateLine(self):
        start = Point(self.source.connectPoint())
        if isinstance(self.target, TerminalGraphicsItem):
            stop = Point(self.target.connectPoint())
        elif isinstance(self.target, QtCore.QPointF):
            stop = Point(self.target)
        else:
            return
        self.prepareGeometryChange()
        self.path = self.generatePath(start, stop)
        self.shapePath = None
        self.update()

    def generatePath(self, start, stop):
        path = QtGui.QPainterPath()
        path.moveTo(start)
        if self.style['shape'] == 'line':
            path.lineTo(stop)
        elif self.style['shape'] == 'cubic':
            path.cubicTo(Point(stop.x(), start.y()), Point(start.x(), stop.y()), Point(stop.x(), stop.y()))
        else:
            raise Exception('Invalid shape "%s"; options are "line" or "cubic"' % self.style['shape'])
        return path

    def keyPressEvent(self, ev):
        if not self.isSelected():
            ev.ignore()
            return
        if ev.key() == QtCore.Qt.Key.Key_Delete or ev.key() == QtCore.Qt.Key.Key_Backspace:
            self.source.disconnect(self.target)
            ev.accept()
        else:
            ev.ignore()

    def mousePressEvent(self, ev):
        ev.ignore()

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            ev.accept()
            sel = self.isSelected()
            self.setSelected(True)
            self.setFocus()
            if not sel and self.isSelected():
                self.update()

    def hoverEvent(self, ev):
        if not ev.isExit() and ev.acceptClicks(QtCore.Qt.MouseButton.LeftButton):
            self.hovered = True
        else:
            self.hovered = False
        self.update()

    def boundingRect(self):
        return self.shape().boundingRect()

    def viewRangeChanged(self):
        self.shapePath = None
        self.prepareGeometryChange()

    def shape(self):
        if self.shapePath is None:
            if self.path is None:
                return QtGui.QPainterPath()
            stroker = QtGui.QPainterPathStroker()
            px = self.pixelWidth()
            stroker.setWidth(px * 8)
            self.shapePath = stroker.createStroke(self.path)
        return self.shapePath

    def paint(self, p, *args):
        if self.isSelected():
            p.setPen(fn.mkPen(self.style['selectedColor'], width=self.style['selectedWidth']))
        elif self.hovered:
            p.setPen(fn.mkPen(self.style['hoverColor'], width=self.style['hoverWidth']))
        else:
            p.setPen(fn.mkPen(self.style['color'], width=self.style['width']))
        p.drawPath(self.path)