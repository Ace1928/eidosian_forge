import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
class ROI(GraphicsObject):
    """
    Generic region-of-interest widget.
    
    Can be used for implementing many types of selection box with 
    rotate/translate/scale handles.
    ROIs can be customized to have a variety of shapes (by subclassing or using
    any of the built-in subclasses) and any combination of draggable handles
    that allow the user to manipulate the ROI.

    Default mouse interaction:

      * Left drag moves the ROI
      * Left drag + Ctrl moves the ROI with position snapping
      * Left drag + Alt rotates the ROI
      * Left drag + Alt + Ctrl rotates the ROI with angle snapping
      * Left drag + Shift scales the ROI
      * Left drag + Shift + Ctrl scales the ROI with size snapping

    In addition to the above interaction modes, it is possible to attach any
    number of handles to the ROI that can be dragged to change the ROI in
    various ways (see the ROI.add____Handle methods).


    ================ ===========================================================
    **Arguments**
    pos              (length-2 sequence) Indicates the position of the ROI's 
                     origin. For most ROIs, this is the lower-left corner of
                     its bounding rectangle.
    size             (length-2 sequence) Indicates the width and height of the 
                     ROI.
    angle            (float) The rotation of the ROI in degrees. Default is 0.
    invertible       (bool) If True, the user may resize the ROI to have 
                     negative width or height (assuming the ROI has scale
                     handles). Default is False.
    maxBounds        (QRect, QRectF, or None) Specifies boundaries that the ROI 
                     cannot be dragged outside of by the user. Default is None.
    snapSize         (float) The spacing of snap positions used when *scaleSnap*
                     or *translateSnap* are enabled. Default is 1.0.
    scaleSnap        (bool) If True, the width and height of the ROI are forced
                     to be integer multiples of *snapSize* when being resized
                     by the user. Default is False.
    translateSnap    (bool) If True, the x and y positions of the ROI are forced
                     to be integer multiples of *snapSize* when being resized
                     by the user. Default is False.
    rotateSnap       (bool) If True, the ROI angle is forced to a multiple of 
                     the ROI's snap angle (default is 15 degrees) when rotated
                     by the user. Default is False.
    parent           (QGraphicsItem) The graphics item parent of this ROI. It
                     is generally not necessary to specify the parent.
    pen              (QPen or argument to pg.mkPen) The pen to use when drawing
                     the shape of the ROI.
    hoverPen         (QPen or argument to mkPen) The pen to use while the
                     mouse is hovering over the ROI shape.
    handlePen        (QPen or argument to mkPen) The pen to use when drawing
                     the ROI handles.
    handleHoverPen   (QPen or argument to mkPen) The pen to use while the mouse
                     is hovering over an ROI handle.
    movable          (bool) If True, the ROI can be moved by dragging anywhere 
                     inside the ROI. Default is True.
    rotatable        (bool) If True, the ROI can be rotated by mouse drag + ALT
    resizable        (bool) If True, the ROI can be resized by mouse drag + 
                     SHIFT
    removable        (bool) If True, the ROI will be given a context menu with
                     an option to remove the ROI. The ROI emits
                     sigRemoveRequested when this menu action is selected.
                     Default is False.
    ================ ===========================================================
    
    
    
    ======================= ====================================================
    **Signals**
    sigRegionChangeFinished Emitted when the user stops dragging the ROI (or
                            one of its handles) or if the ROI is changed
                            programatically.
    sigRegionChangeStarted  Emitted when the user starts dragging the ROI (or
                            one of its handles).
    sigRegionChanged        Emitted any time the position of the ROI changes,
                            including while it is being dragged by the user.
    sigHoverEvent           Emitted when the mouse hovers over the ROI.
    sigClicked              Emitted when the user clicks on the ROI.
                            Note that clicking is disabled by default to prevent
                            stealing clicks from objects behind the ROI. To 
                            enable clicking, call 
                            roi.setAcceptedMouseButtons(QtCore.Qt.MouseButton.LeftButton). 
                            See QtWidgets.QGraphicsItem documentation for more 
                            details.
    sigRemoveRequested      Emitted when the user selects 'remove' from the 
                            ROI's context menu (if available).
    ======================= ====================================================
    """
    sigRegionChangeFinished = QtCore.Signal(object)
    sigRegionChangeStarted = QtCore.Signal(object)
    sigRegionChanged = QtCore.Signal(object)
    sigHoverEvent = QtCore.Signal(object)
    sigClicked = QtCore.Signal(object, object)
    sigRemoveRequested = QtCore.Signal(object)

    def __init__(self, pos, size=Point(1, 1), angle=0.0, invertible=False, maxBounds=None, snapSize=1.0, scaleSnap=False, translateSnap=False, rotateSnap=False, parent=None, pen=None, hoverPen=None, handlePen=None, handleHoverPen=None, movable=True, rotatable=True, resizable=True, removable=False, aspectLocked=False):
        GraphicsObject.__init__(self, parent)
        self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        pos = Point(pos)
        size = Point(size)
        self.aspectLocked = aspectLocked
        self.translatable = movable
        self.rotatable = rotatable
        self.resizable = resizable
        self.removable = removable
        self.menu = None
        self.freeHandleMoved = False
        self.mouseHovering = False
        if pen is None:
            pen = (255, 255, 255)
        self.setPen(pen)
        if hoverPen is None:
            hoverPen = (255, 255, 0)
        self.hoverPen = fn.mkPen(hoverPen)
        if handlePen is None:
            handlePen = (150, 255, 255)
        self.handlePen = fn.mkPen(handlePen)
        if handleHoverPen is None:
            handleHoverPen = (255, 255, 0)
        self.handleHoverPen = handleHoverPen
        self.handles = []
        self.state = {'pos': Point(0, 0), 'size': Point(1, 1), 'angle': 0}
        self.lastState = None
        self.setPos(pos)
        self.setAngle(angle)
        self.setSize(size)
        self.setZValue(10)
        self.isMoving = False
        self.handleSize = 5
        self.invertible = invertible
        self.maxBounds = maxBounds
        self.snapSize = snapSize
        self.translateSnap = translateSnap
        self.rotateSnap = rotateSnap
        self.rotateSnapAngle = 15.0
        self.scaleSnap = scaleSnap
        self.scaleSnapSize = snapSize
        self.mouseDragHandler = MouseDragHandler(self)

    def getState(self):
        return self.stateCopy()

    def stateCopy(self):
        sc = {}
        sc['pos'] = Point(self.state['pos'])
        sc['size'] = Point(self.state['size'])
        sc['angle'] = self.state['angle']
        return sc

    def saveState(self):
        """Return the state of the widget in a format suitable for storing to 
        disk. (Points are converted to tuple)
        
        Combined with setState(), this allows ROIs to be easily saved and 
        restored."""
        state = {}
        state['pos'] = tuple(self.state['pos'])
        state['size'] = tuple(self.state['size'])
        state['angle'] = self.state['angle']
        return state

    def setState(self, state, update=True):
        """
        Set the state of the ROI from a structure generated by saveState() or
        getState().
        """
        self.setPos(state['pos'], update=False)
        self.setSize(state['size'], update=False)
        self.setAngle(state['angle'], update=update)

    def setZValue(self, z):
        QtWidgets.QGraphicsItem.setZValue(self, z)
        for h in self.handles:
            h['item'].setZValue(z + 1)

    def parentBounds(self):
        """
        Return the bounding rectangle of this ROI in the coordinate system
        of its parent.        
        """
        return self.mapToParent(self.boundingRect()).boundingRect()

    def setPen(self, *args, **kwargs):
        """
        Set the pen to use when drawing the ROI shape.
        For arguments, see :func:`mkPen <pyqtgraph.mkPen>`.
        """
        self.pen = fn.mkPen(*args, **kwargs)
        self.currentPen = self.pen
        self.update()

    def size(self):
        """Return the size (w,h) of the ROI."""
        return self.getState()['size']

    def pos(self):
        """Return the position (x,y) of the ROI's origin. 
        For most ROIs, this will be the lower-left corner."""
        return self.getState()['pos']

    def angle(self):
        """Return the angle of the ROI in degrees."""
        return self.getState()['angle']

    def setPos(self, pos, y=None, update=True, finish=True):
        """Set the position of the ROI (in the parent's coordinate system).
        
        Accepts either separate (x, y) arguments or a single :class:`Point` or
        ``QPointF`` argument. 
        
        By default, this method causes both ``sigRegionChanged`` and
        ``sigRegionChangeFinished`` to be emitted. If *finish* is False, then
        ``sigRegionChangeFinished`` will not be emitted. You can then use 
        stateChangeFinished() to cause the signal to be emitted after a series
        of state changes.
        
        If *update* is False, the state change will be remembered but not processed and no signals 
        will be emitted. You can then use stateChanged() to complete the state change. This allows
        multiple change functions to be called sequentially while minimizing processing overhead
        and repeated signals. Setting ``update=False`` also forces ``finish=False``.
        """
        if update not in (True, False):
            raise TypeError('update argument must be bool')
        if y is None:
            pos = Point(pos)
        else:
            if isinstance(y, bool):
                raise TypeError('Positional arguments to setPos() must be numerical.')
            pos = Point(pos, y)
        self.state['pos'] = pos
        QtWidgets.QGraphicsItem.setPos(self, pos)
        if update:
            self.stateChanged(finish=finish)

    def setSize(self, size, center=None, centerLocal=None, snap=False, update=True, finish=True):
        """
        Set the ROI's size.
        
        =============== ==========================================================================
        **Arguments**
        size            (Point | QPointF | sequence) The final size of the ROI
        center          (None | Point) Optional center point around which the ROI is scaled,
                        expressed as [0-1, 0-1] over the size of the ROI.
        centerLocal     (None | Point) Same as *center*, but the position is expressed in the
                        local coordinate system of the ROI
        snap            (bool) If True, the final size is snapped to the nearest increment (see
                        ROI.scaleSnapSize)
        update          (bool) See setPos()
        finish          (bool) See setPos()
        =============== ==========================================================================
        """
        if update not in (True, False):
            raise TypeError('update argument must be bool')
        size = Point(size)
        if snap:
            size[0] = round(size[0] / self.scaleSnapSize) * self.scaleSnapSize
            size[1] = round(size[1] / self.scaleSnapSize) * self.scaleSnapSize
        if centerLocal is not None:
            oldSize = Point(self.state['size'])
            oldSize[0] = 1 if oldSize[0] == 0 else oldSize[0]
            oldSize[1] = 1 if oldSize[1] == 0 else oldSize[1]
            center = Point(centerLocal) / oldSize
        if center is not None:
            center = Point(center)
            c = self.mapToParent(Point(center) * self.state['size'])
            c1 = self.mapToParent(Point(center) * size)
            newPos = self.state['pos'] + c - c1
            self.setPos(newPos, update=False, finish=False)
        self.prepareGeometryChange()
        self.state['size'] = size
        if update:
            self.stateChanged(finish=finish)

    def setAngle(self, angle, center=None, centerLocal=None, snap=False, update=True, finish=True):
        """
        Set the ROI's rotation angle.
        
        =============== ==========================================================================
        **Arguments**
        angle           (float) The final ROI angle in degrees
        center          (None | Point) Optional center point around which the ROI is rotated,
                        expressed as [0-1, 0-1] over the size of the ROI.
        centerLocal     (None | Point) Same as *center*, but the position is expressed in the
                        local coordinate system of the ROI
        snap            (bool) If True, the final ROI angle is snapped to the nearest increment
                        (default is 15 degrees; see ROI.rotateSnapAngle)
        update          (bool) See setPos()
        finish          (bool) See setPos()
        =============== ==========================================================================
        """
        if update not in (True, False):
            raise TypeError('update argument must be bool')
        if snap is True:
            angle = round(angle / self.rotateSnapAngle) * self.rotateSnapAngle
        self.state['angle'] = angle
        tr = QtGui.QTransform()
        tr.rotate(angle)
        if center is not None:
            centerLocal = Point(center) * self.state['size']
        if centerLocal is not None:
            centerLocal = Point(centerLocal)
            cc = self.mapToParent(centerLocal) - (tr.map(centerLocal) + self.state['pos'])
            self.translate(cc, update=False)
        self.setTransform(tr)
        if update:
            self.stateChanged(finish=finish)

    def scale(self, s, center=None, centerLocal=None, snap=False, update=True, finish=True):
        """
        Resize the ROI by scaling relative to *center*.
        See setPos() for an explanation of the *update* and *finish* arguments.
        """
        newSize = self.state['size'] * s
        self.setSize(newSize, center=center, centerLocal=centerLocal, snap=snap, update=update, finish=finish)

    def translate(self, *args, **kargs):
        """
        Move the ROI to a new position.
        Accepts either (x, y, snap) or ([x,y], snap) as arguments
        If the ROI is bounded and the move would exceed boundaries, then the ROI
        is moved to the nearest acceptable position instead.
        
        *snap* can be:
        
        =============== ==========================================================================
        None (default)  use self.translateSnap and self.snapSize to determine whether/how to snap
        False           do not snap
        Point(w,h)      snap to rectangular grid with spacing (w,h)
        True            snap using self.snapSize (and ignoring self.translateSnap)
        =============== ==========================================================================
           
        Also accepts *update* and *finish* arguments (see setPos() for a description of these).
        """
        if len(args) == 1:
            pt = args[0]
        else:
            pt = args
        newState = self.stateCopy()
        newState['pos'] = newState['pos'] + pt
        snap = kargs.get('snap', None)
        if snap is None:
            snap = self.translateSnap
        if snap is not False:
            newState['pos'] = self.getSnapPosition(newState['pos'], snap=snap)
        if self.maxBounds is not None:
            r = self.stateRect(newState)
            d = Point(0, 0)
            if self.maxBounds.left() > r.left():
                d[0] = self.maxBounds.left() - r.left()
            elif self.maxBounds.right() < r.right():
                d[0] = self.maxBounds.right() - r.right()
            if self.maxBounds.top() > r.top():
                d[1] = self.maxBounds.top() - r.top()
            elif self.maxBounds.bottom() < r.bottom():
                d[1] = self.maxBounds.bottom() - r.bottom()
            newState['pos'] += d
        update = kargs.get('update', True)
        finish = kargs.get('finish', True)
        self.setPos(newState['pos'], update=update, finish=finish)

    def rotate(self, angle, center=None, snap=False, update=True, finish=True):
        """
        Rotate the ROI by *angle* degrees. 
        
        =============== ==========================================================================
        **Arguments**
        angle           (float) The angle in degrees to rotate
        center          (None | Point) Optional center point around which the ROI is rotated, in
                        the local coordinate system of the ROI
        snap            (bool) If True, the final ROI angle is snapped to the nearest increment
                        (default is 15 degrees; see ROI.rotateSnapAngle)
        update          (bool) See setPos()
        finish          (bool) See setPos()
        =============== ==========================================================================
        """
        self.setAngle(self.angle() + angle, center=center, snap=snap, update=update, finish=finish)

    def handleMoveStarted(self):
        self.preMoveState = self.getState()
        self.sigRegionChangeStarted.emit(self)

    def addTranslateHandle(self, pos, axes=None, item=None, name=None, index=None):
        """
        Add a new translation handle to the ROI. Dragging the handle will move 
        the entire ROI without changing its angle or shape. 
        
        Note that, by default, ROIs may be moved by dragging anywhere inside the
        ROI. However, for larger ROIs it may be desirable to disable this and
        instead provide one or more translation handles.
        
        =================== ====================================================
        **Arguments**
        pos                 (length-2 sequence) The position of the handle 
                            relative to the shape of the ROI. A value of (0,0)
                            indicates the origin, whereas (1, 1) indicates the
                            upper-right corner, regardless of the ROI's size.
        item                The Handle instance to add. If None, a new handle
                            will be created.
        name                The name of this handle (optional). Handles are 
                            identified by name when calling 
                            getLocalHandlePositions and getSceneHandlePositions.
        =================== ====================================================
        """
        pos = Point(pos)
        return self.addHandle({'name': name, 'type': 't', 'pos': pos, 'item': item}, index=index)

    def addFreeHandle(self, pos=None, axes=None, item=None, name=None, index=None):
        """
        Add a new free handle to the ROI. Dragging free handles has no effect
        on the position or shape of the ROI. 
        
        =================== ====================================================
        **Arguments**
        pos                 (length-2 sequence) The position of the handle 
                            relative to the shape of the ROI. A value of (0,0)
                            indicates the origin, whereas (1, 1) indicates the
                            upper-right corner, regardless of the ROI's size.
        item                The Handle instance to add. If None, a new handle
                            will be created.
        name                The name of this handle (optional). Handles are 
                            identified by name when calling 
                            getLocalHandlePositions and getSceneHandlePositions.
        =================== ====================================================
        """
        if pos is not None:
            pos = Point(pos)
        return self.addHandle({'name': name, 'type': 'f', 'pos': pos, 'item': item}, index=index)

    def addScaleHandle(self, pos, center, axes=None, item=None, name=None, lockAspect=False, index=None):
        """
        Add a new scale handle to the ROI. Dragging a scale handle allows the
        user to change the height and/or width of the ROI.
        
        =================== ====================================================
        **Arguments**
        pos                 (length-2 sequence) The position of the handle 
                            relative to the shape of the ROI. A value of (0,0)
                            indicates the origin, whereas (1, 1) indicates the
                            upper-right corner, regardless of the ROI's size.
        center              (length-2 sequence) The center point around which 
                            scaling takes place. If the center point has the
                            same x or y value as the handle position, then 
                            scaling will be disabled for that axis.
        item                The Handle instance to add. If None, a new handle
                            will be created.
        name                The name of this handle (optional). Handles are 
                            identified by name when calling 
                            getLocalHandlePositions and getSceneHandlePositions.
        =================== ====================================================
        """
        pos = Point(pos)
        center = Point(center)
        info = {'name': name, 'type': 's', 'center': center, 'pos': pos, 'item': item, 'lockAspect': lockAspect}
        if pos.x() == center.x():
            info['xoff'] = True
        if pos.y() == center.y():
            info['yoff'] = True
        return self.addHandle(info, index=index)

    def addRotateHandle(self, pos, center, item=None, name=None, index=None):
        """
        Add a new rotation handle to the ROI. Dragging a rotation handle allows 
        the user to change the angle of the ROI.
        
        =================== ====================================================
        **Arguments**
        pos                 (length-2 sequence) The position of the handle 
                            relative to the shape of the ROI. A value of (0,0)
                            indicates the origin, whereas (1, 1) indicates the
                            upper-right corner, regardless of the ROI's size.
        center              (length-2 sequence) The center point around which 
                            rotation takes place.
        item                The Handle instance to add. If None, a new handle
                            will be created.
        name                The name of this handle (optional). Handles are 
                            identified by name when calling 
                            getLocalHandlePositions and getSceneHandlePositions.
        =================== ====================================================
        """
        pos = Point(pos)
        center = Point(center)
        return self.addHandle({'name': name, 'type': 'r', 'center': center, 'pos': pos, 'item': item}, index=index)

    def addScaleRotateHandle(self, pos, center, item=None, name=None, index=None):
        """
        Add a new scale+rotation handle to the ROI. When dragging a handle of 
        this type, the user can simultaneously rotate the ROI around an 
        arbitrary center point as well as scale the ROI by dragging the handle
        toward or away from the center point.
        
        =================== ====================================================
        **Arguments**
        pos                 (length-2 sequence) The position of the handle 
                            relative to the shape of the ROI. A value of (0,0)
                            indicates the origin, whereas (1, 1) indicates the
                            upper-right corner, regardless of the ROI's size.
        center              (length-2 sequence) The center point around which 
                            scaling and rotation take place.
        item                The Handle instance to add. If None, a new handle
                            will be created.
        name                The name of this handle (optional). Handles are 
                            identified by name when calling 
                            getLocalHandlePositions and getSceneHandlePositions.
        =================== ====================================================
        """
        pos = Point(pos)
        center = Point(center)
        if pos[0] == center[0] and pos[1] == center[1]:
            raise Exception('Scale/rotate handles cannot be at their center point.')
        return self.addHandle({'name': name, 'type': 'sr', 'center': center, 'pos': pos, 'item': item}, index=index)

    def addRotateFreeHandle(self, pos, center, axes=None, item=None, name=None, index=None):
        """
        Add a new rotation+free handle to the ROI. When dragging a handle of 
        this type, the user can rotate the ROI around an 
        arbitrary center point, while moving toward or away from the center 
        point has no effect on the shape of the ROI.
        
        =================== ====================================================
        **Arguments**
        pos                 (length-2 sequence) The position of the handle 
                            relative to the shape of the ROI. A value of (0,0)
                            indicates the origin, whereas (1, 1) indicates the
                            upper-right corner, regardless of the ROI's size.
        center              (length-2 sequence) The center point around which 
                            rotation takes place.
        item                The Handle instance to add. If None, a new handle
                            will be created.
        name                The name of this handle (optional). Handles are 
                            identified by name when calling 
                            getLocalHandlePositions and getSceneHandlePositions.
        =================== ====================================================
        """
        pos = Point(pos)
        center = Point(center)
        return self.addHandle({'name': name, 'type': 'rf', 'center': center, 'pos': pos, 'item': item}, index=index)

    def addHandle(self, info, index=None):
        if 'item' not in info or info['item'] is None:
            h = Handle(self.handleSize, typ=info['type'], pen=self.handlePen, hoverPen=self.handleHoverPen, parent=self)
            info['item'] = h
        else:
            h = info['item']
            if info['pos'] is None:
                info['pos'] = h.pos()
        h.setPos(info['pos'] * self.state['size'])
        h.connectROI(self)
        if index is None:
            self.handles.append(info)
        else:
            self.handles.insert(index, info)
        h.setZValue(self.zValue() + 1)
        self.stateChanged()
        return h

    def indexOfHandle(self, handle):
        """
        Return the index of *handle* in the list of this ROI's handles.
        """
        if isinstance(handle, Handle):
            index = [i for i, info in enumerate(self.handles) if info['item'] is handle]
            if len(index) == 0:
                raise Exception('Cannot return handle index; not attached to this ROI')
            return index[0]
        else:
            return handle

    def removeHandle(self, handle):
        """Remove a handle from this ROI. Argument may be either a Handle 
        instance or the integer index of the handle."""
        index = self.indexOfHandle(handle)
        handle = self.handles[index]['item']
        self.handles.pop(index)
        handle.disconnectROI(self)
        if len(handle.rois) == 0 and self.scene() is not None:
            self.scene().removeItem(handle)
        self.stateChanged()

    def replaceHandle(self, oldHandle, newHandle):
        """Replace one handle in the ROI for another. This is useful when 
        connecting multiple ROIs together.
        
        *oldHandle* may be a Handle instance or the index of a handle to be
        replaced."""
        index = self.indexOfHandle(oldHandle)
        info = self.handles[index]
        self.removeHandle(index)
        info['item'] = newHandle
        info['pos'] = newHandle.pos()
        self.addHandle(info, index=index)

    def checkRemoveHandle(self, handle):
        return True

    def getLocalHandlePositions(self, index=None):
        """Returns the position of handles in the ROI's coordinate system.
        
        The format returned is a list of (name, pos) tuples.
        """
        if index is None:
            positions = []
            for h in self.handles:
                positions.append((h['name'], h['pos']))
            return positions
        else:
            return (self.handles[index]['name'], self.handles[index]['pos'])

    def getSceneHandlePositions(self, index=None):
        """Returns the position of handles in the scene coordinate system.
        
        The format returned is a list of (name, pos) tuples.
        """
        if index is None:
            positions = []
            for h in self.handles:
                positions.append((h['name'], h['item'].scenePos()))
            return positions
        else:
            return (self.handles[index]['name'], self.handles[index]['item'].scenePos())

    def getHandles(self):
        """
        Return a list of this ROI's Handles.
        """
        return [h['item'] for h in self.handles]

    def mapSceneToParent(self, pt):
        return self.mapToParent(self.mapFromScene(pt))

    def setSelected(self, s):
        QtWidgets.QGraphicsItem.setSelected(self, s)
        if s:
            for h in self.handles:
                h['item'].show()
        else:
            for h in self.handles:
                h['item'].hide()

    def hoverEvent(self, ev):
        hover = False
        if not ev.isExit():
            if self.translatable and ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton):
                hover = True
            for btn in [QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.RightButton, QtCore.Qt.MouseButton.MiddleButton]:
                if self.acceptedMouseButtons() & btn and ev.acceptClicks(btn):
                    hover = True
            if self.contextMenuEnabled():
                ev.acceptClicks(QtCore.Qt.MouseButton.RightButton)
        if hover:
            self.setMouseHover(True)
            ev.acceptClicks(QtCore.Qt.MouseButton.LeftButton)
            ev.acceptClicks(QtCore.Qt.MouseButton.RightButton)
            ev.acceptClicks(QtCore.Qt.MouseButton.MiddleButton)
            self.sigHoverEvent.emit(self)
        else:
            self.setMouseHover(False)

    def setMouseHover(self, hover):
        if self.mouseHovering == hover:
            return
        self.mouseHovering = hover
        self._updateHoverColor()

    def _updateHoverColor(self):
        pen = self._makePen()
        if self.currentPen != pen:
            self.currentPen = pen
            self.update()

    def _makePen(self):
        if self.mouseHovering:
            return self.hoverPen
        else:
            return self.pen

    def contextMenuEnabled(self):
        return self.removable or (self.menu and len(self.menu.children()) > 1)

    def raiseContextMenu(self, ev):
        if not self.contextMenuEnabled():
            return
        menu = self.getMenu()
        menu = self.scene().addParentContextMenus(self, menu, ev)
        pos = ev.screenPos()
        menu.popup(QtCore.QPoint(int(pos.x()), int(pos.y())))

    def getMenu(self):
        if self.menu is None:
            self.menu = QtWidgets.QMenu()
            self.menu.setTitle(translate('ROI', 'ROI'))
            if self.removable:
                remAct = QtGui.QAction(translate('ROI', 'Remove ROI'), self.menu)
                remAct.triggered.connect(self.removeClicked)
                self.menu.addAction(remAct)
                self.menu.remAct = remAct
        return self.menu

    def removeClicked(self):
        QtCore.QTimer.singleShot(0, self._emitRemoveRequest)

    def _emitRemoveRequest(self):
        self.sigRemoveRequested.emit(self)

    def mouseDragEvent(self, ev):
        self.mouseDragHandler.mouseDragEvent(ev)

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.RightButton and self.isMoving:
            ev.accept()
            self.cancelMove()
        if ev.button() == QtCore.Qt.MouseButton.RightButton and self.contextMenuEnabled():
            self.raiseContextMenu(ev)
            ev.accept()
        elif self.acceptedMouseButtons() & ev.button():
            ev.accept()
            self.sigClicked.emit(self, ev)
        else:
            ev.ignore()

    def _moveStarted(self):
        self.isMoving = True
        self.preMoveState = self.getState()
        self.sigRegionChangeStarted.emit(self)

    def _moveFinished(self):
        if self.isMoving:
            self.stateChangeFinished()
        self.isMoving = False

    def cancelMove(self):
        self.isMoving = False
        self.setState(self.preMoveState)

    def checkPointMove(self, handle, pos, modifiers):
        """When handles move, they must ask the ROI if the move is acceptable.
        By default, this always returns True. Subclasses may wish override.
        """
        return True

    def movePoint(self, handle, pos, modifiers=None, finish=True, coords='parent'):
        if modifiers is None:
            modifiers = QtCore.Qt.KeyboardModifier.NoModifier
        newState = self.stateCopy()
        index = self.indexOfHandle(handle)
        h = self.handles[index]
        p0 = self.mapToParent(h['pos'] * self.state['size'])
        p1 = Point(pos)
        if coords == 'parent':
            pass
        elif coords == 'scene':
            p1 = self.mapSceneToParent(p1)
        else:
            raise Exception("New point location must be given in either 'parent' or 'scene' coordinates.")
        if 'center' in h:
            c = h['center']
            cs = c * self.state['size']
            lp0 = self.mapFromParent(p0) - cs
            lp1 = self.mapFromParent(p1) - cs
        if h['type'] == 't':
            snap = True if modifiers & QtCore.Qt.KeyboardModifier.ControlModifier else None
            self.translate(p1 - p0, snap=snap, update=False)
        elif h['type'] == 'f':
            newPos = self.mapFromParent(p1)
            h['item'].setPos(newPos)
            h['pos'] = newPos
            self.freeHandleMoved = True
        elif h['type'] == 's':
            if h['center'][0] == h['pos'][0]:
                lp1[0] = 0
            if h['center'][1] == h['pos'][1]:
                lp1[1] = 0
            if self.scaleSnap or modifiers & QtCore.Qt.KeyboardModifier.ControlModifier:
                lp1[0] = round(lp1[0] / self.scaleSnapSize) * self.scaleSnapSize
                lp1[1] = round(lp1[1] / self.scaleSnapSize) * self.scaleSnapSize
            if h['lockAspect'] or modifiers & QtCore.Qt.KeyboardModifier.AltModifier:
                lp1 = lp1.proj(lp0)
            hs = h['pos'] - c
            if hs[0] == 0:
                hs[0] = 1
            if hs[1] == 0:
                hs[1] = 1
            newSize = lp1 / hs
            if newSize[0] == 0:
                newSize[0] = newState['size'][0]
            if newSize[1] == 0:
                newSize[1] = newState['size'][1]
            if not self.invertible:
                if newSize[0] < 0:
                    newSize[0] = newState['size'][0]
                if newSize[1] < 0:
                    newSize[1] = newState['size'][1]
            if self.aspectLocked:
                newSize[0] = newSize[1]
            s0 = c * self.state['size']
            s1 = c * newSize
            cc = self.mapToParent(s0 - s1) - self.mapToParent(Point(0, 0))
            newState['size'] = newSize
            newState['pos'] = newState['pos'] + cc
            if self.maxBounds is not None:
                r = self.stateRect(newState)
                if not self.maxBounds.contains(r):
                    return
            self.setPos(newState['pos'], update=False)
            self.setSize(newState['size'], update=False)
        elif h['type'] in ['r', 'rf']:
            if h['type'] == 'rf':
                self.freeHandleMoved = True
            if not self.rotatable:
                return
            try:
                if lp1.length() == 0 or lp0.length() == 0:
                    return
            except OverflowError:
                return
            ang = newState['angle'] - lp0.angle(lp1)
            if ang is None:
                return
            if self.rotateSnap or modifiers & QtCore.Qt.KeyboardModifier.ControlModifier:
                ang = round(ang / self.rotateSnapAngle) * self.rotateSnapAngle
            tr = QtGui.QTransform()
            tr.rotate(ang)
            cc = self.mapToParent(cs) - (tr.map(cs) + self.state['pos'])
            newState['angle'] = ang
            newState['pos'] = newState['pos'] + cc
            if self.maxBounds is not None:
                r = self.stateRect(newState)
                if not self.maxBounds.contains(r):
                    return
            self.setPos(newState['pos'], update=False)
            self.setAngle(ang, update=False)
            if h['type'] == 'rf':
                h['item'].setPos(self.mapFromScene(p1))
                h['pos'] = self.mapFromParent(p1)
        elif h['type'] == 'sr':
            try:
                if lp1.length() == 0 or lp0.length() == 0:
                    return
            except OverflowError:
                return
            ang = newState['angle'] - lp0.angle(lp1)
            if ang is None:
                return
            if self.rotateSnap or modifiers & QtCore.Qt.KeyboardModifier.ControlModifier:
                ang = round(ang / self.rotateSnapAngle) * self.rotateSnapAngle
            if self.aspectLocked or h['center'][0] != h['pos'][0]:
                newState['size'][0] = self.state['size'][0] * lp1.length() / lp0.length()
                if self.scaleSnap:
                    newState['size'][0] = round(newState['size'][0] / self.snapSize) * self.snapSize
            if self.aspectLocked or h['center'][1] != h['pos'][1]:
                newState['size'][1] = self.state['size'][1] * lp1.length() / lp0.length()
                if self.scaleSnap:
                    newState['size'][1] = round(newState['size'][1] / self.snapSize) * self.snapSize
            if newState['size'][0] == 0:
                newState['size'][0] = 1
            if newState['size'][1] == 0:
                newState['size'][1] = 1
            c1 = c * newState['size']
            tr = QtGui.QTransform()
            tr.rotate(ang)
            cc = self.mapToParent(cs) - (tr.map(c1) + self.state['pos'])
            newState['angle'] = ang
            newState['pos'] = newState['pos'] + cc
            if self.maxBounds is not None:
                r = self.stateRect(newState)
                if not self.maxBounds.contains(r):
                    return
            self.setState(newState, update=False)
        self.stateChanged(finish=finish)

    def stateChanged(self, finish=True):
        """Process changes to the state of the ROI.
        If there are any changes, then the positions of handles are updated accordingly
        and sigRegionChanged is emitted. If finish is True, then 
        sigRegionChangeFinished will also be emitted."""
        changed = False
        if self.lastState is None:
            changed = True
        else:
            state = self.getState()
            for k in list(state.keys()):
                if state[k] != self.lastState[k]:
                    changed = True
        self.prepareGeometryChange()
        if changed:
            for h in self.handles:
                if h['item'] in self.childItems():
                    h['item'].setPos(h['pos'] * self.state['size'])
            self.update()
            self.sigRegionChanged.emit(self)
        elif self.freeHandleMoved:
            self.sigRegionChanged.emit(self)
        self.freeHandleMoved = False
        self.lastState = self.getState()
        if finish:
            self.stateChangeFinished()
            self.informViewBoundsChanged()

    def stateChangeFinished(self):
        self.sigRegionChangeFinished.emit(self)

    def stateRect(self, state):
        r = QtCore.QRectF(0, 0, state['size'][0], state['size'][1])
        tr = QtGui.QTransform()
        tr.rotate(-state['angle'])
        r = tr.mapRect(r)
        return r.adjusted(state['pos'][0], state['pos'][1], state['pos'][0], state['pos'][1])

    def getSnapPosition(self, pos, snap=None):
        if snap is None or snap is True:
            if self.snapSize is None:
                return pos
            snap = Point(self.snapSize, self.snapSize)
        return Point(round(pos[0] / snap[0]) * snap[0], round(pos[1] / snap[1]) * snap[1])

    def boundingRect(self):
        return QtCore.QRectF(0, 0, self.state['size'][0], self.state['size'][1]).normalized()

    def paint(self, p, opt, widget):
        r = QtCore.QRectF(0, 0, self.state['size'][0], self.state['size'][1]).normalized()
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setPen(self.currentPen)
        p.translate(r.left(), r.top())
        p.scale(r.width(), r.height())
        p.drawRect(0, 0, 1, 1)

    def getArraySlice(self, data, img, axes=(0, 1), returnSlice=True):
        """Return a tuple of slice objects that can be used to slice the region
        from *data* that is covered by the bounding rectangle of this ROI.
        Also returns the transform that maps the ROI into data coordinates.
        
        If returnSlice is set to False, the function returns a pair of tuples with the values that would have 
        been used to generate the slice objects. ((ax0Start, ax0Stop), (ax1Start, ax1Stop))
        
        If the slice cannot be computed (usually because the scene/transforms are not properly
        constructed yet), then the method returns None.
        """
        dShape = (data.shape[axes[0]], data.shape[axes[1]])
        try:
            tr = self.sceneTransform() * fn.invertQTransform(img.sceneTransform())
        except np.linalg.linalg.LinAlgError:
            return None
        axisOrder = img.axisOrder
        if axisOrder == 'row-major':
            tr.scale(float(dShape[1]) / img.width(), float(dShape[0]) / img.height())
        else:
            tr.scale(float(dShape[0]) / img.width(), float(dShape[1]) / img.height())
        dataBounds = tr.mapRect(self.boundingRect())
        if axisOrder == 'row-major':
            intBounds = dataBounds.intersected(QtCore.QRectF(0, 0, dShape[1], dShape[0]))
        else:
            intBounds = dataBounds.intersected(QtCore.QRectF(0, 0, dShape[0], dShape[1]))
        bounds = ((int(min(intBounds.left(), intBounds.right())), int(1 + max(intBounds.left(), intBounds.right()))), (int(min(intBounds.bottom(), intBounds.top())), int(1 + max(intBounds.bottom(), intBounds.top()))))
        if axisOrder == 'row-major':
            bounds = bounds[::-1]
        if returnSlice:
            sl = [slice(None)] * data.ndim
            sl[axes[0]] = slice(*bounds[0])
            sl[axes[1]] = slice(*bounds[1])
            return (tuple(sl), tr)
        else:
            return (bounds, tr)

    def getArrayRegion(self, data, img, axes=(0, 1), returnMappedCoords=False, **kwds):
        """Use the position and orientation of this ROI relative to an imageItem
        to pull a slice from an array.

        =================== ====================================================
        **Arguments**
        data                The array to slice from. Note that this array does
                            *not* have to be the same data that is represented
                            in *img*.
        img                 (ImageItem or other suitable QGraphicsItem)
                            Used to determine the relationship between the 
                            ROI and the boundaries of *data*.
        axes                (length-2 tuple) Specifies the axes in *data* that
                            correspond to the (x, y) axes of *img*. If the
                            image's axis order is set to
                            'row-major', then the axes are instead specified in
                            (y, x) order.
        returnMappedCoords  (bool) If True, the array slice is returned along
                            with a corresponding array of coordinates that were
                            used to extract data from the original array.
        \\**kwds             All keyword arguments are passed to 
                            :func:`affineSlice <pyqtgraph.affineSlice>`.
        =================== ====================================================
        
        This method uses :func:`affineSlice <pyqtgraph.affineSlice>` to generate
        the slice from *data* and uses :func:`getAffineSliceParams <pyqtgraph.ROI.getAffineSliceParams>`
        to determine the parameters to pass to :func:`affineSlice <pyqtgraph.affineSlice>`.
        
        If *returnMappedCoords* is True, then the method returns a tuple (result, coords) 
        such that coords is the set of coordinates used to interpolate values from the original
        data, mapped into the parent coordinate system of the image. This is useful, when slicing
        data from images that have been transformed, for determining the location of each value
        in the sliced data.
        
        All extra keyword arguments are passed to :func:`affineSlice <pyqtgraph.affineSlice>`.
        """
        fromBR = kwds.pop('fromBoundingRect', False)
        _shape, _vectors, _origin = self.getAffineSliceParams(data, img, axes, fromBoundingRect=fromBR)
        shape = kwds.pop('shape', _shape)
        vectors = kwds.pop('vectors', _vectors)
        origin = kwds.pop('origin', _origin)
        if not returnMappedCoords:
            rgn = fn.affineSlice(data, shape=shape, vectors=vectors, origin=origin, axes=axes, **kwds)
            return rgn
        else:
            kwds['returnCoords'] = True
            result, coords = fn.affineSlice(data, shape=shape, vectors=vectors, origin=origin, axes=axes, **kwds)
            mapped = fn.transformCoordinates(img.transform(), coords)
            return (result, mapped)

    def _getArrayRegionForArbitraryShape(self, data, img, axes=(0, 1), returnMappedCoords=False, **kwds):
        """
        Return the result of :meth:`~pyqtgraph.ROI.getArrayRegion`, masked by
        the shape of the ROI. Values outside the ROI shape are set to 0.

        See :meth:`~pyqtgraph.ROI.getArrayRegion` for a description of the
        arguments.
        """
        if returnMappedCoords:
            sliced, mappedCoords = ROI.getArrayRegion(self, data, img, axes, returnMappedCoords, fromBoundingRect=True, **kwds)
        else:
            sliced = ROI.getArrayRegion(self, data, img, axes, returnMappedCoords, fromBoundingRect=True, **kwds)
        if img.axisOrder == 'col-major':
            mask = self.renderShapeMask(sliced.shape[axes[0]], sliced.shape[axes[1]])
        else:
            mask = self.renderShapeMask(sliced.shape[axes[1]], sliced.shape[axes[0]])
            mask = mask.T
        shape = [1] * data.ndim
        shape[axes[0]] = sliced.shape[axes[0]]
        shape[axes[1]] = sliced.shape[axes[1]]
        mask = mask.reshape(shape)
        if returnMappedCoords:
            return (sliced * mask, mappedCoords)
        else:
            return sliced * mask

    def getAffineSliceParams(self, data, img, axes=(0, 1), fromBoundingRect=False):
        """
        Returns the parameters needed to use :func:`affineSlice <pyqtgraph.affineSlice>`
        (shape, vectors, origin) to extract a subset of *data* using this ROI 
        and *img* to specify the subset.
        
        If *fromBoundingRect* is True, then the ROI's bounding rectangle is used
        rather than the shape of the ROI.
        
        See :func:`getArrayRegion <pyqtgraph.ROI.getArrayRegion>` for more information.
        """
        if self.scene() is not img.scene():
            raise Exception('ROI and target item must be members of the same scene.')
        origin = img.mapToData(self.mapToItem(img, QtCore.QPointF(0, 0)))
        vx = img.mapToData(self.mapToItem(img, QtCore.QPointF(1, 0))) - origin
        vy = img.mapToData(self.mapToItem(img, QtCore.QPointF(0, 1))) - origin
        lvx = hypot(vx.x(), vx.y())
        lvy = hypot(vy.x(), vy.y())
        sx = 1.0 / lvx
        sy = 1.0 / lvy
        vectors = ((vx.x() * sx, vx.y() * sx), (vy.x() * sy, vy.y() * sy))
        if fromBoundingRect is True:
            shape = (self.boundingRect().width(), self.boundingRect().height())
            origin = img.mapToData(self.mapToItem(img, self.boundingRect().topLeft()))
            origin = (origin.x(), origin.y())
        else:
            shape = self.state['size']
            origin = (origin.x(), origin.y())
        shape = [abs(shape[0] / sx), abs(shape[1] / sy)]
        if img.axisOrder == 'row-major':
            vectors = vectors[::-1]
            shape = shape[::-1]
        return (shape, vectors, origin)

    def renderShapeMask(self, width, height):
        """Return an array of 0.0-1.0 into which the shape of the item has been drawn.
        
        This can be used to mask array selections.
        """
        if width == 0 or height == 0:
            return np.empty((width, height), dtype=float)
        im = QtGui.QImage(width, height, QtGui.QImage.Format.Format_ARGB32)
        im.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(im)
        p.setPen(fn.mkPen(None))
        p.setBrush(fn.mkBrush('w'))
        shape = self.shape()
        bounds = shape.boundingRect()
        p.scale(im.width() / bounds.width(), im.height() / bounds.height())
        p.translate(-bounds.topLeft())
        p.drawPath(shape)
        p.end()
        cidx = 0 if sys.byteorder == 'little' else 3
        mask = fn.ndarray_from_qimage(im)[..., cidx].T
        return mask.astype(float) / 255

    def getGlobalTransform(self, relativeTo=None):
        """Return global transformation (rotation angle+translation) required to move 
        from relative state to current state. If relative state isn't specified,
        then we use the state of the ROI when mouse is pressed."""
        if relativeTo is None:
            relativeTo = self.preMoveState
        st = self.getState()
        relativeTo['scale'] = relativeTo['size']
        st['scale'] = st['size']
        t1 = SRTTransform(relativeTo)
        t2 = SRTTransform(st)
        return t2 / t1

    def applyGlobalTransform(self, tr):
        st = self.getState()
        st['scale'] = st['size']
        st = SRTTransform(st)
        st = (st * tr).saveState()
        st['size'] = st['scale']
        self.setState(st)