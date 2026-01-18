import warnings
import weakref
from time import perf_counter, perf_counter_ns
from .. import debug as debug
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets, isQObjectAlive
from .mouseEvents import HoverEvent, MouseClickEvent, MouseDragEvent
class GraphicsScene(QtWidgets.QGraphicsScene):
    """
    Extension of QGraphicsScene that implements a complete, parallel mouse event system.
    (It would have been preferred to just alter the way QGraphicsScene creates and delivers 
    events, but this turned out to be impossible because the constructor for QGraphicsMouseEvent
    is private)
    
      *  Generates MouseClicked events in addition to the usual press/move/release events.
         (This works around a problem where it is impossible to have one item respond to a
         drag if another is watching for a click.)
      *  Adjustable radius around click that will catch objects so you don't have to click *exactly* over small/thin objects
      *  Global context menu--if an item implements a context menu, then its parent(s) may also add items to the menu.
      *  Allows items to decide _before_ a mouse click which item will be the recipient of mouse events.
         This lets us indicate unambiguously to the user which item they are about to click/drag on
      *  Eats mouseMove events that occur too soon after a mouse press.
      *  Reimplements items() and itemAt() to circumvent PyQt bug

    ====================== ====================================================================
    **Signals**
    sigMouseClicked(event) Emitted when the mouse is clicked over the scene. Use ev.pos() to
                           get the click position relative to the item that was clicked on,
                           or ev.scenePos() to get the click position in scene coordinates.
                           See :class:`pyqtgraph.GraphicsScene.mouseEvents.MouseClickEvent`.                        
    sigMouseMoved(pos)     Emitted when the mouse cursor moves over the scene. The position
                           is given in scene coordinates.
    sigMouseHover(items)   Emitted when the mouse is moved over the scene. Items is a list
                           of items under the cursor.
    sigItemAdded(item)     Emitted when an item is added via addItem(). The item is given.
    sigItemRemoved(item)   Emitted when an item is removed via removeItem(). The item is given.
    ====================== ====================================================================
    
    Mouse interaction is as follows:
    
    1) Every time the mouse moves, the scene delivers both the standard hoverEnter/Move/LeaveEvents 
       as well as custom HoverEvents. 
    2) Items are sent HoverEvents in Z-order and each item may optionally call event.acceptClicks(button), 
       acceptDrags(button) or both. If this method call returns True, this informs the item that _if_ 
       the user clicks/drags the specified mouse button, the item is guaranteed to be the 
       recipient of click/drag events (the item may wish to change its appearance to indicate this).
       If the call to acceptClicks/Drags returns False, then the item is guaranteed to *not* receive
       the requested event (because another item has already accepted it). 
    3) If the mouse is clicked, a mousePressEvent is generated as usual. If any items accept this press event, then
       No click/drag events will be generated and mouse interaction proceeds as defined by Qt. This allows
       items to function properly if they are expecting the usual press/move/release sequence of events.
       (It is recommended that items do NOT accept press events, and instead use click/drag events)
       Note: The default implementation of QGraphicsItem.mousePressEvent will *accept* the event if the 
       item is has its Selectable or Movable flags enabled. You may need to override this behavior.
    4) If no item accepts the mousePressEvent, then the scene will begin delivering mouseDrag and/or mouseClick events.
       If the mouse is moved a sufficient distance (or moved slowly enough) before the button is released, 
       then a mouseDragEvent is generated.
       If no drag events are generated before the button is released, then a mouseClickEvent is generated. 
    5) Click/drag events are delivered to the item that called acceptClicks/acceptDrags on the HoverEvent
       in step 1. If no such items exist, then the scene attempts to deliver the events to items near the event. 
       ClickEvents may be delivered in this way even if no
       item originally claimed it could accept the click. DragEvents may only be delivered this way if it is the initial
       move in a drag.
    """
    sigMouseHover = QtCore.Signal(object)
    sigMouseMoved = QtCore.Signal(object)
    sigMouseClicked = QtCore.Signal(object)
    sigPrepareForPaint = QtCore.Signal()
    sigItemAdded = QtCore.Signal(object)
    sigItemRemoved = QtCore.Signal(object)
    _addressCache = weakref.WeakValueDictionary()
    ExportDirectory = None

    def __init__(self, clickRadius: int=2, moveDistance=5, parent=None):
        QtWidgets.QGraphicsScene.__init__(self, parent)
        self.setClickRadius(clickRadius)
        self.setMoveDistance(moveDistance)
        self.exportDirectory = None
        self.clickEvents = []
        self.dragButtons = []
        self.mouseGrabber = None
        self.dragItem = None
        self.lastDrag = None
        self.hoverItems = weakref.WeakKeyDictionary()
        self.lastHoverEvent = None
        self.minDragTime = 0.5
        self.contextMenu = [QtGui.QAction(QtCore.QCoreApplication.translate('GraphicsScene', 'Export...'), self)]
        self.contextMenu[0].triggered.connect(self.showExportDialog)
        self.exportDialog = None
        self._lastMoveEventTime = 0

    def render(self, *args):
        self.prepareForPaint()
        return QtWidgets.QGraphicsScene.render(self, *args)

    def prepareForPaint(self):
        """Called before every render. This method will inform items that the scene is about to
        be rendered by emitting sigPrepareForPaint.
        
        This allows items to delay expensive processing until they know a paint will be required."""
        self.sigPrepareForPaint.emit()

    def setClickRadius(self, r: int):
        """
        Set the distance away from mouse clicks to search for interacting items.
        When clicking, the scene searches first for items that directly intersect the click position
        followed by any other items that are within a rectangle that extends r pixels away from the 
        click position. 
        """
        self._clickRadius = int(r)

    def setMoveDistance(self, d):
        """
        Set the distance the mouse must move after a press before mouseMoveEvents will be delivered.
        This ensures that clicks with a small amount of movement are recognized as clicks instead of
        drags.
        """
        self._moveDistance = d

    def mousePressEvent(self, ev):
        super().mousePressEvent(ev)
        if self.mouseGrabberItem() is None:
            if self.lastHoverEvent is not None:
                if ev.scenePos() != self.lastHoverEvent.scenePos():
                    self.sendHoverEvents(ev)
            self.clickEvents.append(MouseClickEvent(ev))
            items = self.items(ev.scenePos())
            for i in items:
                if i.isEnabled() and i.isVisible() and i.flags() & i.GraphicsItemFlag.ItemIsFocusable:
                    i.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
                    break

    def _moveEventIsAllowed(self):
        rateLimit = getConfigOption('mouseRateLimit')
        if rateLimit <= 0:
            return True
        delay = 1000.0 / rateLimit
        if getMillis() - self._lastMoveEventTime >= delay:
            return True
        return False

    def mouseMoveEvent(self, ev):
        if self._moveEventIsAllowed():
            self._lastMoveEventTime = getMillis()
            self.sigMouseMoved.emit(ev.scenePos())
            super().mouseMoveEvent(ev)
            self.sendHoverEvents(ev)
            if ev.buttons():
                super().mouseMoveEvent(ev)
                if self.mouseGrabberItem() is None:
                    now = perf_counter()
                    init = False
                    for btn in [QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.MiddleButton, QtCore.Qt.MouseButton.RightButton]:
                        if not ev.buttons() & btn:
                            continue
                        if btn not in self.dragButtons:
                            cev = [e for e in self.clickEvents if e.button() == btn]
                            if cev:
                                cev = cev[0]
                                dist = Point(ev.scenePos() - cev.scenePos()).length()
                                if dist == 0 or (dist < self._moveDistance and now - cev.time() < self.minDragTime):
                                    continue
                                init = init or len(self.dragButtons) == 0
                                self.dragButtons.append(btn)
                    if len(self.dragButtons) > 0:
                        if self.sendDragEvent(ev, init=init):
                            ev.accept()
        else:
            super().mouseMoveEvent(ev)
            ev.accept()

    def leaveEvent(self, ev):
        if len(self.dragButtons) == 0:
            self.sendHoverEvents(ev, exitOnly=True)

    def mouseReleaseEvent(self, ev):
        if self.mouseGrabberItem() is None:
            if ev.button() in self.dragButtons:
                if self.sendDragEvent(ev, final=True):
                    ev.accept()
                self.dragButtons.remove(ev.button())
            else:
                cev = [e for e in self.clickEvents if e.button() == ev.button()]
                if cev:
                    if self.sendClickEvent(cev[0]):
                        ev.accept()
                    try:
                        self.clickEvents.remove(cev[0])
                    except ValueError:
                        warnings.warn('A ValueError can occur here with errant QApplication.processEvent() calls, see https://github.com/pyqtgraph/pyqtgraph/pull/2580 for more information.', RuntimeWarning, stacklevel=2)
        if not ev.buttons():
            self.dragItem = None
            self.dragButtons = []
            self.clickEvents = []
            self.lastDrag = None
        super().mouseReleaseEvent(ev)
        self.sendHoverEvents(ev)

    def mouseDoubleClickEvent(self, ev):
        super().mouseDoubleClickEvent(ev)
        if self.mouseGrabberItem() is None:
            self.clickEvents.append(MouseClickEvent(ev, double=True))

    def sendHoverEvents(self, ev, exitOnly=False):
        if exitOnly:
            acceptable = False
            items = []
            event = HoverEvent(None, acceptable)
        else:
            acceptable = not ev.buttons()
            event = HoverEvent(ev, acceptable)
            items = self.itemsNearEvent(event, hoverable=True)
            self.sigMouseHover.emit(items)
        prevItems = list(self.hoverItems.keys())
        for item in items:
            if hasattr(item, 'hoverEvent'):
                event.currentItem = item
                if item not in self.hoverItems:
                    self.hoverItems[item] = None
                    event.enter = True
                else:
                    prevItems.remove(item)
                    event.enter = False
                try:
                    item.hoverEvent(event)
                except:
                    debug.printExc('Error sending hover event:')
        event.enter = False
        event.exit = True
        for item in prevItems:
            event.currentItem = item
            try:
                if isQObjectAlive(item) and item.scene() is self:
                    item.hoverEvent(event)
            except:
                debug.printExc('Error sending hover exit event:')
            finally:
                del self.hoverItems[item]
        if ev.type() == ev.Type.GraphicsSceneMousePress or (ev.type() == ev.Type.GraphicsSceneMouseMove and (not ev.buttons())):
            self.lastHoverEvent = event

    def sendDragEvent(self, ev, init=False, final=False):
        event = MouseDragEvent(ev, self.clickEvents[0], self.lastDrag, start=init, finish=final)
        if init and self.dragItem is None:
            if self.lastHoverEvent is not None:
                acceptedItem = self.lastHoverEvent.dragItems().get(event.button(), None)
            else:
                acceptedItem = None
            if acceptedItem is not None and acceptedItem.scene() is self:
                self.dragItem = acceptedItem
                event.currentItem = self.dragItem
                try:
                    self.dragItem.mouseDragEvent(event)
                except:
                    debug.printExc('Error sending drag event:')
            else:
                for item in self.itemsNearEvent(event):
                    if not item.isVisible() or not item.isEnabled():
                        continue
                    if hasattr(item, 'mouseDragEvent'):
                        event.currentItem = item
                        try:
                            item.mouseDragEvent(event)
                        except:
                            debug.printExc('Error sending drag event:')
                        if event.isAccepted():
                            self.dragItem = item
                            if item.flags() & item.GraphicsItemFlag.ItemIsFocusable:
                                item.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
                            break
        elif self.dragItem is not None:
            event.currentItem = self.dragItem
            try:
                self.dragItem.mouseDragEvent(event)
            except:
                debug.printExc('Error sending hover exit event:')
        self.lastDrag = event
        return event.isAccepted()

    def sendClickEvent(self, ev):
        if self.dragItem is not None and hasattr(self.dragItem, 'mouseClickEvent'):
            ev.currentItem = self.dragItem
            self.dragItem.mouseClickEvent(ev)
        else:
            if self.lastHoverEvent is not None:
                acceptedItem = self.lastHoverEvent.clickItems().get(ev.button(), None)
            else:
                acceptedItem = None
            if acceptedItem is not None:
                ev.currentItem = acceptedItem
                try:
                    acceptedItem.mouseClickEvent(ev)
                except:
                    debug.printExc('Error sending click event:')
            else:
                for item in self.itemsNearEvent(ev):
                    if not item.isVisible() or not item.isEnabled():
                        continue
                    if hasattr(item, 'mouseClickEvent'):
                        ev.currentItem = item
                        try:
                            item.mouseClickEvent(ev)
                        except:
                            debug.printExc('Error sending click event:')
                        if ev.isAccepted():
                            if item.flags() & item.GraphicsItemFlag.ItemIsFocusable:
                                item.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
                            break
        self.sigMouseClicked.emit(ev)
        return ev.isAccepted()

    def addItem(self, item):
        ret = QtWidgets.QGraphicsScene.addItem(self, item)
        self.sigItemAdded.emit(item)
        return ret

    def removeItem(self, item):
        ret = QtWidgets.QGraphicsScene.removeItem(self, item)
        self.sigItemRemoved.emit(item)
        return ret

    def itemsNearEvent(self, event, selMode=QtCore.Qt.ItemSelectionMode.IntersectsItemShape, sortOrder=QtCore.Qt.SortOrder.DescendingOrder, hoverable=False):
        """
        Return an iterator that iterates first through the items that directly intersect point (in Z order)
        followed by any other items that are within the scene's click radius.
        """
        view = self.views()[0]
        tr = view.viewportTransform()
        if hasattr(event, 'buttonDownScenePos'):
            point = event.buttonDownScenePos()
        else:
            point = event.scenePos()

        def absZValue(item):
            if item is None:
                return 0
            return item.zValue() + absZValue(item.parentItem())
        items_at_point = self.items(point, selMode, sortOrder, tr)
        items_at_point.sort(key=absZValue, reverse=True)
        r = self._clickRadius
        items_within_radius = []
        rgn = None
        if r > 0:
            rect = view.mapToScene(QtCore.QRect(0, 0, 2 * r, 2 * r)).boundingRect()
            w = rect.width()
            h = rect.height()
            rgn = QtCore.QRectF(point.x() - w / 2, point.y() - h / 2, w, h)
            items_within_radius = self.items(rgn, selMode, sortOrder, tr)
            items_within_radius.sort(key=absZValue, reverse=True)
            for item in items_at_point:
                if item in items_within_radius:
                    items_within_radius.remove(item)
        all_items = items_at_point + items_within_radius
        selected_items = []
        for item in all_items:
            if hoverable and (not hasattr(item, 'hoverEvent')):
                continue
            if item.scene() is not self:
                continue
            shape = item.shape()
            if shape is None:
                continue
            if rgn is not None and shape.intersects(item.mapFromScene(rgn).boundingRect()) or shape.contains(item.mapFromScene(point)):
                selected_items.append(item)
        return selected_items

    def getViewWidget(self):
        return self.views()[0]

    def addParentContextMenus(self, item, menu, event):
        """
        Can be called by any item in the scene to expand its context menu to include parent context menus.
        Parents may implement getContextMenus to add new menus / actions to the existing menu.
        getContextMenus must accept 1 argument (the event that generated the original menu) and
        return a single QMenu or a list of QMenus.
        
        The final menu will look like:
        
            |    Original Item 1
            |    Original Item 2
            |    ...
            |    Original Item N
            |    ------------------
            |    Parent Item 1
            |    Parent Item 2
            |    ...
            |    Grandparent Item 1
            |    ...
            
        
        ==============  ==================================================
        **Arguments:**
        item            The item that initially created the context menu 
                        (This is probably the item making the call to this function)
        menu            The context menu being shown by the item
        event           The original event that triggered the menu to appear.
        ==============  ==================================================
        """
        menusToAdd = []
        while item is not self:
            item = item.parentItem()
            if item is None:
                item = self
            if not hasattr(item, 'getContextMenus'):
                continue
            subMenus = item.getContextMenus(event) or []
            if isinstance(subMenus, list):
                menusToAdd.extend(subMenus)
            else:
                menusToAdd.append(subMenus)
        existingActions = menu.actions()
        actsToAdd = []
        for menuOrAct in menusToAdd:
            if isinstance(menuOrAct, QtWidgets.QMenu):
                menuOrAct = menuOrAct.menuAction()
            elif not isinstance(menuOrAct, QtGui.QAction):
                raise Exception(f'Cannot add object {menuOrAct} (type={type(menuOrAct)}) to QMenu.')
            if menuOrAct not in existingActions:
                actsToAdd.append(menuOrAct)
        if actsToAdd:
            menu.addSeparator()
        menu.addActions(actsToAdd)
        return menu

    def getContextMenus(self, event):
        self.contextMenuItem = event.acceptedItem
        return self.contextMenu

    def showExportDialog(self):
        if self.exportDialog is None:
            from . import exportDialog
            self.exportDialog = exportDialog.ExportDialog(self)
        self.exportDialog.show(self.contextMenuItem)