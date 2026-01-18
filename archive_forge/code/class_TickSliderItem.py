import operator
import weakref
import numpy as np
from .. import functions as fn
from .. import colormap
from ..colormap import ColorMap
from ..Qt import QtCore, QtGui, QtWidgets
from ..widgets.SpinBox import SpinBox
from ..widgets.ColorMapButton import ColorMapMenu
from .GraphicsWidget import GraphicsWidget
from .GradientPresets import Gradients
class TickSliderItem(GraphicsWidget):
    """**Bases:** :class:`GraphicsWidget <pyqtgraph.GraphicsWidget>`
    
    A rectangular item with tick marks along its length that can (optionally) be moved by the user."""
    sigTicksChanged = QtCore.Signal(object)
    sigTicksChangeFinished = QtCore.Signal(object)

    def __init__(self, orientation='bottom', allowAdd=True, allowRemove=True, **kargs):
        """
        ==============  =================================================================================
        **Arguments:**
        orientation     Set the orientation of the gradient. Options are: 'left', 'right'
                        'top', and 'bottom'.
        allowAdd        Specifies whether the user can add ticks.
        allowRemove     Specifies whether the user can remove new ticks.
        tickPen         Default is white. Specifies the color of the outline of the ticks.
                        Can be any of the valid arguments for :func:`mkPen <pyqtgraph.mkPen>`
        ==============  =================================================================================
        """
        GraphicsWidget.__init__(self)
        self.orientation = orientation
        self.length = 100
        self.tickSize = 15
        self.ticks = {}
        self.maxDim = 20
        self.allowAdd = allowAdd
        self.allowRemove = allowRemove
        if 'tickPen' in kargs:
            self.tickPen = fn.mkPen(kargs['tickPen'])
        else:
            self.tickPen = fn.mkPen('w')
        self.orientations = {'left': (90, 1, 1), 'right': (90, 1, 1), 'top': (0, 1, -1), 'bottom': (0, 1, 1)}
        self.setOrientation(orientation)

    def paint(self, p, opt, widget):
        return

    def keyPressEvent(self, ev):
        ev.ignore()

    def setMaxDim(self, mx=None):
        if mx is None:
            mx = self.maxDim
        else:
            self.maxDim = mx
        if self.orientation in ['bottom', 'top']:
            self.setFixedHeight(mx)
            self.setMaximumWidth(16777215)
        else:
            self.setFixedWidth(mx)
            self.setMaximumHeight(16777215)

    def setOrientation(self, orientation):
        """Set the orientation of the TickSliderItem.
        
        ==============  ===================================================================
        **Arguments:**
        orientation     Options are: 'left', 'right', 'top', 'bottom'
                        The orientation option specifies which side of the slider the
                        ticks are on, as well as whether the slider is vertical ('right'
                        and 'left') or horizontal ('top' and 'bottom').
        ==============  ===================================================================
        """
        self.orientation = orientation
        self.setMaxDim()
        self.resetTransform()
        ort = orientation
        if ort == 'top':
            transform = QtGui.QTransform.fromScale(1, -1)
            transform.translate(0, -self.height())
            self.setTransform(transform)
        elif ort == 'left':
            transform = QtGui.QTransform()
            transform.rotate(270)
            transform.scale(1, -1)
            transform.translate(-self.height(), -self.maxDim)
            self.setTransform(transform)
        elif ort == 'right':
            transform = QtGui.QTransform()
            transform.rotate(270)
            transform.translate(-self.height(), 0)
            self.setTransform(transform)
        elif ort != 'bottom':
            raise Exception("%s is not a valid orientation. Options are 'left', 'right', 'top', and 'bottom'" % str(ort))
        tr = QtGui.QTransform.fromTranslate(self.tickSize / 2.0, 0)
        self.setTransform(tr, True)

    def addTick(self, x, color=None, movable=True, finish=True):
        """
        Add a tick to the item.
        
        ==============  ==================================================================
        **Arguments:**
        x               Position where tick should be added.
        color           Color of added tick. If color is not specified, the color will be
                        white.
        movable         Specifies whether the tick is movable with the mouse.
        ==============  ==================================================================
        """
        if color is None:
            color = QtGui.QColor(255, 255, 255)
        tick = Tick([x * self.length, 0], color, movable, self.tickSize, pen=self.tickPen, removeAllowed=self.allowRemove)
        self.ticks[tick] = x
        tick.setParentItem(self)
        tick.sigMoving.connect(self.tickMoved)
        tick.sigMoved.connect(self.tickMoveFinished)
        tick.sigClicked.connect(self.tickClicked)
        self.sigTicksChanged.emit(self)
        if finish:
            self.sigTicksChangeFinished.emit(self)
        return tick

    def removeTick(self, tick, finish=True):
        """
        Removes the specified tick.
        """
        del self.ticks[tick]
        tick.setParentItem(None)
        if self.scene() is not None:
            self.scene().removeItem(tick)
        self.sigTicksChanged.emit(self)
        if finish:
            self.sigTicksChangeFinished.emit(self)

    def tickMoved(self, tick, pos):
        newX = min(max(0, pos.x()), self.length)
        pos.setX(newX)
        tick.setPos(pos)
        self.ticks[tick] = float(newX) / self.length
        self.sigTicksChanged.emit(self)

    def tickMoveFinished(self, tick):
        self.sigTicksChangeFinished.emit(self)

    def tickClicked(self, tick, ev):
        if ev.button() == QtCore.Qt.MouseButton.RightButton and tick.removeAllowed:
            self.removeTick(tick)

    def widgetLength(self):
        if self.orientation in ['bottom', 'top']:
            return self.width()
        else:
            return self.height()

    def resizeEvent(self, ev):
        wlen = max(40, self.widgetLength())
        self.setLength(wlen - self.tickSize - 2)
        self.setOrientation(self.orientation)

    def setLength(self, newLen):
        for t, x in list(self.ticks.items()):
            t.setPos(x * newLen + 1, t.pos().y())
        self.length = float(newLen)

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton and self.allowAdd:
            pos = ev.pos()
            if pos.x() < 0 or pos.x() > self.length:
                return
            if pos.y() < 0 or pos.y() > self.tickSize:
                return
            pos.setX(min(max(pos.x(), 0), self.length))
            self.addTick(pos.x() / self.length)
        elif ev.button() == QtCore.Qt.MouseButton.RightButton:
            self.showMenu(ev)

    def hoverEvent(self, ev):
        if not ev.isExit() and ev.acceptClicks(QtCore.Qt.MouseButton.LeftButton):
            ev.acceptClicks(QtCore.Qt.MouseButton.RightButton)

    def showMenu(self, ev):
        pass

    def setTickColor(self, tick, color):
        """Set the color of the specified tick.
        
        ==============  ==================================================================
        **Arguments:**
        tick            Can be either an integer corresponding to the index of the tick
                        or a Tick object. Ex: if you had a slider with 3 ticks and you
                        wanted to change the middle tick, the index would be 1.
        color           The color to make the tick. Can be any argument that is valid for
                        :func:`mkBrush <pyqtgraph.mkBrush>`
        ==============  ==================================================================
        """
        tick = self.getTick(tick)
        tick.color = color
        tick.update()
        self.sigTicksChanged.emit(self)
        self.sigTicksChangeFinished.emit(self)

    def setTickValue(self, tick, val):
        """
        Set the position (along the slider) of the tick.
        
        ==============   ==================================================================
        **Arguments:**
        tick             Can be either an integer corresponding to the index of the tick
                         or a Tick object. Ex: if you had a slider with 3 ticks and you
                         wanted to change the middle tick, the index would be 1.
        val              The desired position of the tick. If val is < 0, position will be
                         set to 0. If val is > 1, position will be set to 1.
        ==============   ==================================================================
        """
        tick = self.getTick(tick)
        val = min(max(0.0, val), 1.0)
        x = val * self.length
        pos = tick.pos()
        pos.setX(x)
        tick.setPos(pos)
        self.ticks[tick] = val
        self.update()
        self.sigTicksChanged.emit(self)
        self.sigTicksChangeFinished.emit(self)

    def tickValue(self, tick):
        """Return the value (from 0.0 to 1.0) of the specified tick.
        
        ==============  ==================================================================
        **Arguments:**
        tick            Can be either an integer corresponding to the index of the tick
                        or a Tick object. Ex: if you had a slider with 3 ticks and you
                        wanted the value of the middle tick, the index would be 1.
        ==============  ==================================================================
        """
        tick = self.getTick(tick)
        return self.ticks[tick]

    def getTick(self, tick):
        """Return the Tick object at the specified index.
        
        ==============  ==================================================================
        **Arguments:**
        tick            An integer corresponding to the index of the desired tick. If the
                        argument is not an integer it will be returned unchanged.
        ==============  ==================================================================
        """
        if type(tick) is int:
            tick = self.listTicks()[tick][0]
        return tick

    def listTicks(self):
        """Return a sorted list of all the Tick objects on the slider."""
        ticks = sorted(self.ticks.items(), key=operator.itemgetter(1))
        return ticks