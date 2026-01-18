import weakref
from time import perf_counter
from ..Point import Point
from ..Qt import QtCore
class MouseDragEvent(object):
    """
    Instances of this class are delivered to items in a :class:`GraphicsScene <pyqtgraph.GraphicsScene>` via their mouseDragEvent() method when the item is being mouse-dragged. 
    
    """

    def __init__(self, moveEvent, pressEvent, lastEvent, start=False, finish=False):
        self.start = start
        self.finish = finish
        self.accepted = False
        self.currentItem = None
        self._buttonDownScenePos = {}
        self._buttonDownScreenPos = {}
        for btn in [QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.MiddleButton, QtCore.Qt.MouseButton.RightButton]:
            self._buttonDownScenePos[btn] = moveEvent.buttonDownScenePos(btn)
            self._buttonDownScreenPos[btn] = moveEvent.buttonDownScreenPos(btn)
        self._scenePos = moveEvent.scenePos()
        self._screenPos = moveEvent.screenPos()
        if lastEvent is None:
            self._lastScenePos = pressEvent.scenePos()
            self._lastScreenPos = pressEvent.screenPos()
        else:
            self._lastScenePos = lastEvent.scenePos()
            self._lastScreenPos = lastEvent.screenPos()
        self._buttons = moveEvent.buttons()
        self._button = pressEvent.button()
        self._modifiers = moveEvent.modifiers()
        self.acceptedItem = None

    def accept(self):
        """An item should call this method if it can handle the event. This will prevent the event being delivered to any other items."""
        self.accepted = True
        self.acceptedItem = self.currentItem

    def ignore(self):
        """An item should call this method if it cannot handle the event. This will allow the event to be delivered to other items."""
        self.accepted = False

    def isAccepted(self):
        return self.accepted

    def scenePos(self):
        """Return the current scene position of the mouse."""
        return Point(self._scenePos)

    def screenPos(self):
        """Return the current screen position (pixels relative to widget) of the mouse."""
        return Point(self._screenPos)

    def buttonDownScenePos(self, btn=None):
        """
        Return the scene position of the mouse at the time *btn* was pressed.
        If *btn* is omitted, then the button that initiated the drag is assumed.
        """
        if btn is None:
            btn = self.button()
        return Point(self._buttonDownScenePos[btn])

    def buttonDownScreenPos(self, btn=None):
        """
        Return the screen position (pixels relative to widget) of the mouse at the time *btn* was pressed.
        If *btn* is omitted, then the button that initiated the drag is assumed.
        """
        if btn is None:
            btn = self.button()
        return Point(self._buttonDownScreenPos[btn])

    def lastScenePos(self):
        """
        Return the scene position of the mouse immediately prior to this event.
        """
        return Point(self._lastScenePos)

    def lastScreenPos(self):
        """
        Return the screen position of the mouse immediately prior to this event.
        """
        return Point(self._lastScreenPos)

    def buttons(self):
        """
        Return the buttons currently pressed on the mouse.
        (see QGraphicsSceneMouseEvent::buttons in the Qt documentation)
        """
        return self._buttons

    def button(self):
        """Return the button that initiated the drag (may be different from the buttons currently pressed)
        (see QGraphicsSceneMouseEvent::button in the Qt documentation)
        
        """
        return self._button

    def pos(self):
        """
        Return the current position of the mouse in the coordinate system of the item
        that the event was delivered to.
        """
        return Point(self.currentItem.mapFromScene(self._scenePos))

    def lastPos(self):
        """
        Return the previous position of the mouse in the coordinate system of the item
        that the event was delivered to.
        """
        return Point(self.currentItem.mapFromScene(self._lastScenePos))

    def buttonDownPos(self, btn=None):
        """
        Return the position of the mouse at the time the drag was initiated
        in the coordinate system of the item that the event was delivered to.
        """
        if btn is None:
            btn = self.button()
        return Point(self.currentItem.mapFromScene(self._buttonDownScenePos[btn]))

    def isStart(self):
        """Returns True if this event is the first since a drag was initiated."""
        return self.start

    def isFinish(self):
        """Returns False if this is the last event in a drag. Note that this
        event will have the same position as the previous one."""
        return self.finish

    def __repr__(self):
        if self.currentItem is None:
            lp = self._lastScenePos
            p = self._scenePos
        else:
            lp = self.lastPos()
            p = self.pos()
        return '<MouseDragEvent (%g,%g)->(%g,%g) buttons=%s start=%s finish=%s>' % (lp.x(), lp.y(), p.x(), p.y(), str(self.buttons()), str(self.isStart()), str(self.isFinish()))

    def modifiers(self):
        """Return any keyboard modifiers currently pressed.
        (see QGraphicsSceneMouseEvent::modifiers in the Qt documentation)
        
        """
        return self._modifiers