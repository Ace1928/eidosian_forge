from math import atan2, degrees
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui
from .GraphicsItem import GraphicsItem
from .GraphicsObject import GraphicsObject
from .TextItem import TextItem
from .ViewBox import ViewBox
class InfLineLabel(TextItem):
    """
    A TextItem that attaches itself to an InfiniteLine.
    
    This class extends TextItem with the following features:
    
      * Automatically positions adjacent to the line at a fixed position along
        the line and within the view box.
      * Automatically reformats text when the line value has changed.
      * Can optionally be dragged to change its location along the line.
      * Optionally aligns to its parent line.

    =============== ==================================================================
    **Arguments:**
    line            The InfiniteLine to which this label will be attached.
    text            String to display in the label. May contain a {value} formatting
                    string to display the current value of the line.
    movable         Bool; if True, then the label can be dragged along the line.
    position        Relative position (0.0-1.0) within the view to position the label
                    along the line.
    anchors         List of (x,y) pairs giving the text anchor positions that should
                    be used when the line is moved to one side of the view or the
                    other. This allows text to switch to the opposite side of the line
                    as it approaches the edge of the view. These are automatically
                    selected for some common cases, but may be specified if the 
                    default values give unexpected results.
    =============== ==================================================================
    
    All extra keyword arguments are passed to TextItem. A particularly useful
    option here is to use `rotateAxis=(1, 0)`, which will cause the text to
    be automatically rotated parallel to the line.
    """

    def __init__(self, line, text='', movable=False, position=0.5, anchors=None, **kwds):
        self.line = line
        self.movable = movable
        self.moving = False
        self.orthoPos = position
        self.format = text
        self.line.sigPositionChanged.connect(self.valueChanged)
        self._endpoints = (None, None)
        if anchors is None:
            rax = kwds.get('rotateAxis', None)
            if rax is not None:
                if tuple(rax) == (1, 0):
                    anchors = [(0.5, 0), (0.5, 1)]
                else:
                    anchors = [(0, 0.5), (1, 0.5)]
            elif line.angle % 180 == 0:
                anchors = [(0.5, 0), (0.5, 1)]
            else:
                anchors = [(0, 0.5), (1, 0.5)]
        self.anchors = anchors
        TextItem.__init__(self, **kwds)
        self.setParentItem(line)
        self.valueChanged()

    def valueChanged(self):
        if not self.isVisible():
            return
        value = self.line.value()
        self.setText(self.format.format(value=value))
        self.updatePosition()

    def getEndpoints(self):
        if self._endpoints[0] is None:
            lr = self.line.boundingRect()
            pt1 = Point(lr.left(), 0)
            pt2 = Point(lr.right(), 0)
            if self.line.angle % 90 != 0:
                view = self.getViewBox()
                if not self.isVisible() or not isinstance(view, ViewBox):
                    return (None, None)
                p = QtGui.QPainterPath()
                p.moveTo(pt1)
                p.lineTo(pt2)
                p = self.line.itemTransform(view)[0].map(p)
                vr = QtGui.QPainterPath()
                vr.addRect(view.boundingRect())
                paths = vr.intersected(p).toSubpathPolygons(QtGui.QTransform())
                if len(paths) > 0:
                    l = list(paths[0])
                    pt1 = self.line.mapFromItem(view, l[0])
                    pt2 = self.line.mapFromItem(view, l[1])
            self._endpoints = (pt1, pt2)
        return self._endpoints

    def updatePosition(self):
        self._endpoints = (None, None)
        pt1, pt2 = self.getEndpoints()
        if pt1 is None:
            return
        pt = pt2 * self.orthoPos + pt1 * (1 - self.orthoPos)
        self.setPos(pt)
        vr = self.line.viewRect()
        if vr is not None:
            self.setAnchor(self.anchors[0 if vr.center().y() < 0 else 1])

    def setVisible(self, v):
        TextItem.setVisible(self, v)
        if v:
            self.valueChanged()

    def setMovable(self, m):
        """Set whether this label is movable by dragging along the line.
        """
        self.movable = m
        self.setAcceptHoverEvents(m)

    def setPosition(self, p):
        """Set the relative position (0.0-1.0) of this label within the view box
        and along the line. 
        
        For horizontal (angle=0) and vertical (angle=90) lines, a value of 0.0
        places the text at the bottom or left of the view, respectively. 
        """
        self.orthoPos = p
        self.updatePosition()

    def setFormat(self, text):
        """Set the text format string for this label.
        
        May optionally contain "{value}" to include the lines current value
        (the text will be reformatted whenever the line is moved).
        """
        self.format = text
        self.valueChanged()

    def mouseDragEvent(self, ev):
        if self.movable and ev.button() == QtCore.Qt.MouseButton.LeftButton:
            if ev.isStart():
                self._moving = True
                self._cursorOffset = self._posToRel(ev.buttonDownPos())
                self._startPosition = self.orthoPos
            ev.accept()
            if not self._moving:
                return
            rel = self._posToRel(ev.pos())
            self.orthoPos = fn.clip_scalar(self._startPosition + rel - self._cursorOffset, 0.0, 1.0)
            self.updatePosition()
            if ev.isFinish():
                self._moving = False

    def mouseClickEvent(self, ev):
        if self.moving and ev.button() == QtCore.Qt.MouseButton.RightButton:
            ev.accept()
            self.orthoPos = self._startPosition
            self.moving = False

    def hoverEvent(self, ev):
        if not ev.isExit() and self.movable:
            ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton)

    def viewTransformChanged(self):
        GraphicsItem.viewTransformChanged(self)
        self.updatePosition()
        TextItem.viewTransformChanged(self)

    def _posToRel(self, pos):
        pt1, pt2 = self.getEndpoints()
        if pt1 is None:
            return 0
        pos = self.mapToParent(pos)
        return (pos.x() - pt1.x()) / (pt2.x() - pt1.x())