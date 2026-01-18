import math
from PySide2 import QtCore, QtGui, QtWidgets
import diagramscene_rc
class DiagramItem(QtWidgets.QGraphicsPolygonItem):
    Step, Conditional, StartEnd, Io = range(4)

    def __init__(self, diagramType, contextMenu, parent=None, scene=None):
        super(DiagramItem, self).__init__(parent, scene)
        self.arrows = []
        self.diagramType = diagramType
        self.myContextMenu = contextMenu
        path = QtGui.QPainterPath()
        if self.diagramType == self.StartEnd:
            path.moveTo(200, 50)
            path.arcTo(150, 0, 50, 50, 0, 90)
            path.arcTo(50, 0, 50, 50, 90, 90)
            path.arcTo(50, 50, 50, 50, 180, 90)
            path.arcTo(150, 50, 50, 50, 270, 90)
            path.lineTo(200, 25)
            self.myPolygon = path.toFillPolygon()
        elif self.diagramType == self.Conditional:
            self.myPolygon = QtGui.QPolygonF([QtCore.QPointF(-100, 0), QtCore.QPointF(0, 100), QtCore.QPointF(100, 0), QtCore.QPointF(0, -100), QtCore.QPointF(-100, 0)])
        elif self.diagramType == self.Step:
            self.myPolygon = QtGui.QPolygonF([QtCore.QPointF(-100, -100), QtCore.QPointF(100, -100), QtCore.QPointF(100, 100), QtCore.QPointF(-100, 100), QtCore.QPointF(-100, -100)])
        else:
            self.myPolygon = QtGui.QPolygonF([QtCore.QPointF(-120, -80), QtCore.QPointF(-70, 80), QtCore.QPointF(120, 80), QtCore.QPointF(70, -80), QtCore.QPointF(-120, -80)])
        self.setPolygon(self.myPolygon)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)

    def removeArrow(self, arrow):
        try:
            self.arrows.remove(arrow)
        except ValueError:
            pass

    def removeArrows(self):
        for arrow in self.arrows[:]:
            arrow.startItem().removeArrow(arrow)
            arrow.endItem().removeArrow(arrow)
            self.scene().removeItem(arrow)

    def addArrow(self, arrow):
        self.arrows.append(arrow)

    def image(self):
        pixmap = QtGui.QPixmap(250, 250)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setPen(QtGui.QPen(QtCore.Qt.black, 8))
        painter.translate(125, 125)
        painter.drawPolyline(self.myPolygon)
        return pixmap

    def contextMenuEvent(self, event):
        self.scene().clearSelection()
        self.setSelected(True)
        self.myContextMenu.exec_(event.screenPos())

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionChange:
            for arrow in self.arrows:
                arrow.updatePosition()
        return value