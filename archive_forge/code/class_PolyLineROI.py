import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
class PolyLineROI(ROI):
    """
    Container class for multiple connected LineSegmentROIs.

    This class allows the user to draw paths of multiple line segments.

    ============== =============================================================
    **Arguments**
    positions      (list of length-2 sequences) The list of points in the path.
                   Note that, unlike the handle positions specified in other
                   ROIs, these positions must be expressed in the normal
                   coordinate system of the ROI, rather than (0 to 1) relative
                   to the size of the ROI.
    closed         (bool) if True, an extra LineSegmentROI is added connecting 
                   the beginning and end points.
    \\**args        All extra keyword arguments are passed to ROI()
    ============== =============================================================
    
    """

    def __init__(self, positions, closed=False, pos=None, **args):
        if pos is None:
            pos = [0, 0]
        self.closed = closed
        self.segments = []
        ROI.__init__(self, pos, size=[1, 1], **args)
        self.setPoints(positions)

    def setPoints(self, points, closed=None):
        """
        Set the complete sequence of points displayed by this ROI.
        
        ============= =========================================================
        **Arguments**
        points        List of (x,y) tuples specifying handle locations to set.
        closed        If bool, then this will set whether the ROI is closed 
                      (the last point is connected to the first point). If
                      None, then the closed mode is left unchanged.
        ============= =========================================================
        
        """
        if closed is not None:
            self.closed = closed
        self.clearPoints()
        for p in points:
            self.addFreeHandle(p)
        start = -1 if self.closed else 0
        for i in range(start, len(self.handles) - 1):
            self.addSegment(self.handles[i]['item'], self.handles[i + 1]['item'])

    def clearPoints(self):
        """
        Remove all handles and segments.
        """
        while len(self.handles) > 0:
            self.removeHandle(self.handles[0]['item'])

    def getState(self):
        state = ROI.getState(self)
        state['closed'] = self.closed
        state['points'] = [Point(h.pos()) for h in self.getHandles()]
        return state

    def saveState(self):
        state = ROI.saveState(self)
        state['closed'] = self.closed
        state['points'] = [tuple(h.pos()) for h in self.getHandles()]
        return state

    def setState(self, state):
        ROI.setState(self, state)
        self.setPoints(state['points'], closed=state['closed'])

    def addSegment(self, h1, h2, index=None):
        seg = _PolyLineSegment(handles=(h1, h2), pen=self.pen, hoverPen=self.hoverPen, parent=self, movable=False)
        if index is None:
            self.segments.append(seg)
        else:
            self.segments.insert(index, seg)
        seg.sigClicked.connect(self.segmentClicked)
        seg.setAcceptedMouseButtons(QtCore.Qt.MouseButton.LeftButton)
        seg.setZValue(self.zValue() + 1)
        for h in seg.handles:
            h['item'].setDeletable(True)
            h['item'].setAcceptedMouseButtons(h['item'].acceptedMouseButtons() | QtCore.Qt.MouseButton.LeftButton)

    def setMouseHover(self, hover):
        ROI.setMouseHover(self, hover)
        for s in self.segments:
            s.setParentHover(hover)

    def addHandle(self, info, index=None):
        h = ROI.addHandle(self, info, index=index)
        h.sigRemoveRequested.connect(self.removeHandle)
        self.stateChanged(finish=True)
        return h

    def segmentClicked(self, segment, ev=None, pos=None):
        if ev is not None:
            pos = segment.mapToParent(ev.pos())
        elif pos is None:
            raise Exception('Either an event or a position must be given.')
        h2 = segment.handles[1]['item']
        i = self.segments.index(segment)
        h3 = self.addFreeHandle(pos, index=self.indexOfHandle(h2))
        self.addSegment(h3, h2, index=i + 1)
        segment.replaceHandle(h2, h3)

    def removeHandle(self, handle, updateSegments=True):
        ROI.removeHandle(self, handle)
        handle.sigRemoveRequested.disconnect(self.removeHandle)
        if not updateSegments:
            return
        segments = handle.rois[:]
        if len(segments) == 1:
            self.removeSegment(segments[0])
        elif len(segments) > 1:
            handles = [h['item'] for h in segments[1].handles]
            handles.remove(handle)
            segments[0].replaceHandle(handle, handles[0])
            self.removeSegment(segments[1])
        self.stateChanged(finish=True)

    def removeSegment(self, seg):
        for handle in seg.handles[:]:
            seg.removeHandle(handle['item'])
        self.segments.remove(seg)
        seg.sigClicked.disconnect(self.segmentClicked)
        self.scene().removeItem(seg)

    def checkRemoveHandle(self, h):
        if self.closed:
            return len(self.handles) > 3
        else:
            return len(self.handles) > 2

    def paint(self, p, *args):
        pass

    def boundingRect(self):
        return self.shape().boundingRect()

    def shape(self):
        p = QtGui.QPainterPath()
        if len(self.handles) == 0:
            return p
        p.moveTo(self.handles[0]['item'].pos())
        for i in range(len(self.handles)):
            p.lineTo(self.handles[i]['item'].pos())
        p.lineTo(self.handles[0]['item'].pos())
        return p

    def getArrayRegion(self, *args, **kwds):
        return self._getArrayRegionForArbitraryShape(*args, **kwds)

    def setPen(self, *args, **kwds):
        ROI.setPen(self, *args, **kwds)
        for seg in self.segments:
            seg.setPen(*args, **kwds)