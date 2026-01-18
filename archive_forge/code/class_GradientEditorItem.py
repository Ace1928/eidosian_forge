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
class GradientEditorItem(TickSliderItem):
    """
    **Bases:** :class:`TickSliderItem <pyqtgraph.TickSliderItem>`
    
    An item that can be used to define a color gradient. Implements common pre-defined gradients that are 
    customizable by the user. :class: `GradientWidget <pyqtgraph.GradientWidget>` provides a widget
    with a GradientEditorItem that can be added to a GUI. 
    
    ================================ ===========================================================
    **Signals:**
    sigGradientChanged(self)         Signal is emitted anytime the gradient changes. The signal 
                                     is emitted in real time while ticks are being dragged or 
                                     colors are being changed.
    sigGradientChangeFinished(self)  Signal is emitted when the gradient is finished changing.
    ================================ ===========================================================    
 
    """
    sigGradientChanged = QtCore.Signal(object)
    sigGradientChangeFinished = QtCore.Signal(object)

    def __init__(self, *args, **kargs):
        """
        Create a new GradientEditorItem. 
        All arguments are passed to :func:`TickSliderItem.__init__ <pyqtgraph.TickSliderItem.__init__>`
        
        ===============  =================================================================================
        **Arguments:**
        orientation      Set the orientation of the gradient. Options are: 'left', 'right'
                         'top', and 'bottom'.
        allowAdd         Default is True. Specifies whether ticks can be added to the item.
        tickPen          Default is white. Specifies the color of the outline of the ticks.
                         Can be any of the valid arguments for :func:`mkPen <pyqtgraph.mkPen>`
        ===============  =================================================================================
        """
        self.currentTick = None
        self.currentTickColor = None
        self.rectSize = 15
        self.gradRect = QtWidgets.QGraphicsRectItem(QtCore.QRectF(0, self.rectSize, 100, self.rectSize))
        self.backgroundRect = QtWidgets.QGraphicsRectItem(QtCore.QRectF(0, -self.rectSize, 100, self.rectSize))
        self.backgroundRect.setBrush(QtGui.QBrush(QtCore.Qt.BrushStyle.DiagCrossPattern))
        self.colorMode = 'rgb'
        TickSliderItem.__init__(self, *args, **kargs)
        self.colorDialog = QtWidgets.QColorDialog()
        self.colorDialog.setOption(QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel, True)
        self.colorDialog.setOption(QtWidgets.QColorDialog.ColorDialogOption.DontUseNativeDialog, True)
        self.colorDialog.currentColorChanged.connect(self.currentColorChanged)
        self.colorDialog.rejected.connect(self.currentColorRejected)
        self.colorDialog.accepted.connect(self.currentColorAccepted)
        self.backgroundRect.setParentItem(self)
        self.gradRect.setParentItem(self)
        self.setMaxDim(self.rectSize + self.tickSize)
        self.rgbAction = QtGui.QAction(translate('GradiantEditorItem', 'RGB'), self)
        self.rgbAction.setCheckable(True)
        self.rgbAction.triggered.connect(self._setColorModeToRGB)
        self.hsvAction = QtGui.QAction(translate('GradiantEditorItem', 'HSV'), self)
        self.hsvAction.setCheckable(True)
        self.hsvAction.triggered.connect(self._setColorModeToHSV)
        self.menu = ColorMapMenu(showGradientSubMenu=True)
        self.menu.triggered.connect(self.contextMenuClicked)
        self.menu.addSeparator()
        self.menu.addAction(self.rgbAction)
        self.menu.addAction(self.hsvAction)
        for t in list(self.ticks.keys()):
            self.removeTick(t)
        self.addTick(0, QtGui.QColor(0, 0, 0), True)
        self.addTick(1, QtGui.QColor(255, 0, 0), True)
        self.setColorMode('rgb')
        self.updateGradient()
        self.linkedGradients = {}
        self.sigTicksChanged.connect(self._updateGradientIgnoreArgs)
        self.sigTicksChangeFinished.connect(self.sigGradientChangeFinished)

    def showTicks(self, show=True):
        for tick in self.ticks.keys():
            if show:
                tick.show()
                orig = getattr(self, '_allowAdd_backup', None)
                if orig:
                    self.allowAdd = orig
            else:
                self._allowAdd_backup = self.allowAdd
                self.allowAdd = False
                tick.hide()

    def setOrientation(self, orientation):
        """
        Set the orientation of the GradientEditorItem. 
        
        ==============  ===================================================================
        **Arguments:**
        orientation     Options are: 'left', 'right', 'top', 'bottom'
                        The orientation option specifies which side of the gradient the
                        ticks are on, as well as whether the gradient is vertical ('right'
                        and 'left') or horizontal ('top' and 'bottom').
        ==============  ===================================================================
        """
        TickSliderItem.setOrientation(self, orientation)
        tr = QtGui.QTransform.fromTranslate(0, self.rectSize)
        self.setTransform(tr, True)

    def showMenu(self, ev):
        self.menu.popup(ev.screenPos().toQPoint())

    def contextMenuClicked(self, action):
        if action in [self.rgbAction, self.hsvAction]:
            return
        name, source = action.data()
        if source == 'preset-gradient':
            self.loadPreset(name)
        else:
            if name is None:
                cmap = colormap.ColorMap(None, [0.0, 1.0])
            else:
                cmap = colormap.get(name, source=source)
            self.setColorMap(cmap)
            self.showTicks(False)

    @addGradientListToDocstring()
    def loadPreset(self, name):
        """
        Load a predefined gradient. Currently defined gradients are: 
        """
        self.restoreState(Gradients[name])

    def setColorMode(self, cm):
        """
        Set the color mode for the gradient. Options are: 'hsv', 'rgb'
        
        """
        if cm not in ['rgb', 'hsv']:
            raise Exception("Unknown color mode %s. Options are 'rgb' and 'hsv'." % str(cm))
        try:
            self.rgbAction.blockSignals(True)
            self.hsvAction.blockSignals(True)
            self.rgbAction.setChecked(cm == 'rgb')
            self.hsvAction.setChecked(cm == 'hsv')
        finally:
            self.rgbAction.blockSignals(False)
            self.hsvAction.blockSignals(False)
        self.colorMode = cm
        self.sigTicksChanged.emit(self)
        self.sigGradientChangeFinished.emit(self)

    def _setColorModeToRGB(self):
        self.setColorMode('rgb')

    def _setColorModeToHSV(self):
        self.setColorMode('hsv')

    def colorMap(self):
        """Return a ColorMap object representing the current state of the editor."""
        if self.colorMode == 'hsv':
            raise NotImplementedError('hsv colormaps not yet supported')
        pos = []
        color = []
        for t, x in self.listTicks():
            pos.append(x)
            c = t.color
            color.append(c.getRgb())
        return ColorMap(np.array(pos), np.array(color, dtype=np.ubyte))

    def updateGradient(self):
        self.gradient = self.getGradient()
        self.gradRect.setBrush(QtGui.QBrush(self.gradient))
        self.sigGradientChanged.emit(self)

    def _updateGradientIgnoreArgs(self, *args, **kwargs):
        self.updateGradient()

    def setLength(self, newLen):
        TickSliderItem.setLength(self, newLen)
        self.backgroundRect.setRect(1, -self.rectSize, newLen, self.rectSize)
        self.gradRect.setRect(1, -self.rectSize, newLen, self.rectSize)
        self.sigTicksChanged.emit(self)

    def currentColorChanged(self, color):
        if color.isValid() and self.currentTick is not None:
            self.setTickColor(self.currentTick, color)

    def currentColorRejected(self):
        self.setTickColor(self.currentTick, self.currentTickColor)

    def currentColorAccepted(self):
        self.sigGradientChangeFinished.emit(self)

    def tickClicked(self, tick, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self.raiseColorDialog(tick)
        elif ev.button() == QtCore.Qt.MouseButton.RightButton:
            self.raiseTickContextMenu(tick, ev)

    def raiseColorDialog(self, tick):
        if not tick.colorChangeAllowed:
            return
        self.currentTick = tick
        self.currentTickColor = tick.color
        self.colorDialog.setCurrentColor(tick.color)
        self.colorDialog.open()

    def raiseTickContextMenu(self, tick, ev):
        self.tickMenu = TickMenu(tick, self)
        self.tickMenu.popup(ev.screenPos().toQPoint())

    def tickMoveFinished(self, tick):
        self.sigGradientChangeFinished.emit(self)

    def getGradient(self):
        """Return a QLinearGradient object."""
        g = QtGui.QLinearGradient(QtCore.QPointF(0, 0), QtCore.QPointF(self.length, 0))
        if self.colorMode == 'rgb':
            ticks = self.listTicks()
            g.setStops([(x, QtGui.QColor(t.color)) for t, x in ticks])
        elif self.colorMode == 'hsv':
            ticks = self.listTicks()
            stops = []
            stops.append((ticks[0][1], ticks[0][0].color))
            for i in range(1, len(ticks)):
                x1 = ticks[i - 1][1]
                x2 = ticks[i][1]
                dx = (x2 - x1) / 10.0
                for j in range(1, 10):
                    x = x1 + dx * j
                    stops.append((x, self.getColor(x)))
                stops.append((x2, self.getColor(x2)))
            g.setStops(stops)
        return g

    def getColor(self, x, toQColor=True):
        """
        Return a color for a given value.
        
        ==============  ==================================================================
        **Arguments:**
        x               Value (position on gradient) of requested color.
        toQColor        If true, returns a QColor object, else returns a (r,g,b,a) tuple.
        ==============  ==================================================================
        """
        ticks = self.listTicks()
        if x <= ticks[0][1]:
            c = ticks[0][0].color
            if toQColor:
                return QtGui.QColor(c)
            else:
                return c.getRgb()
        if x >= ticks[-1][1]:
            c = ticks[-1][0].color
            if toQColor:
                return QtGui.QColor(c)
            else:
                return c.getRgb()
        x2 = ticks[0][1]
        for i in range(1, len(ticks)):
            x1 = x2
            x2 = ticks[i][1]
            if x1 <= x and x2 >= x:
                break
        dx = x2 - x1
        if dx == 0:
            f = 0.0
        else:
            f = (x - x1) / dx
        c1 = ticks[i - 1][0].color
        c2 = ticks[i][0].color
        if self.colorMode == 'rgb':
            r = c1.red() * (1.0 - f) + c2.red() * f
            g = c1.green() * (1.0 - f) + c2.green() * f
            b = c1.blue() * (1.0 - f) + c2.blue() * f
            a = c1.alpha() * (1.0 - f) + c2.alpha() * f
            if toQColor:
                return QtGui.QColor(int(r), int(g), int(b), int(a))
            else:
                return (r, g, b, a)
        elif self.colorMode == 'hsv':
            h1, s1, v1, _ = c1.getHsv()
            h2, s2, v2, _ = c2.getHsv()
            h = h1 * (1.0 - f) + h2 * f
            s = s1 * (1.0 - f) + s2 * f
            v = v1 * (1.0 - f) + v2 * f
            c = QtGui.QColor.fromHsv(int(h), int(s), int(v))
            if toQColor:
                return c
            else:
                return c.getRgb()

    def getLookupTable(self, nPts, alpha=None):
        """
        Return an RGB(A) lookup table (ndarray). 
        
        ==============  ============================================================================
        **Arguments:**
        nPts            The number of points in the returned lookup table.
        alpha           True, False, or None - Specifies whether or not alpha values are included
                        in the table.If alpha is None, alpha will be automatically determined.
        ==============  ============================================================================
        """
        if alpha is None:
            alpha = self.usesAlpha()
        if alpha:
            table = np.empty((nPts, 4), dtype=np.ubyte)
        else:
            table = np.empty((nPts, 3), dtype=np.ubyte)
        for i in range(nPts):
            x = float(i) / (nPts - 1)
            color = self.getColor(x, toQColor=False)
            table[i] = color[:table.shape[1]]
        return table

    def usesAlpha(self):
        """Return True if any ticks have an alpha < 255"""
        ticks = self.listTicks()
        for t in ticks:
            if t[0].color.alpha() < 255:
                return True
        return False

    def isLookupTrivial(self):
        """Return True if the gradient has exactly two stops in it: black at 0.0 and white at 1.0"""
        ticks = self.listTicks()
        if len(ticks) != 2:
            return False
        if ticks[0][1] != 0.0 or ticks[1][1] != 1.0:
            return False
        c1 = ticks[0][0].color.getRgb()
        c2 = ticks[1][0].color.getRgb()
        if c1 != (0, 0, 0, 255) or c2 != (255, 255, 255, 255):
            return False
        return True

    def addTick(self, x, color=None, movable=True, finish=True):
        """
        Add a tick to the gradient. Return the tick.
        
        ==============  ==================================================================
        **Arguments:**
        x               Position where tick should be added.
        color           Color of added tick. If color is not specified, the color will be
                        the color of the gradient at the specified position.
        movable         Specifies whether the tick is movable with the mouse.
        ==============  ==================================================================
        """
        if color is None:
            color = self.getColor(x)
        t = TickSliderItem.addTick(self, x, color=color, movable=movable, finish=finish)
        t.colorChangeAllowed = True
        return t

    def saveState(self):
        """
        Return a dictionary with parameters for rebuilding the gradient. Keys will include:
        
           - 'mode': hsv or rgb
           - 'ticks': a list of tuples (pos, (r,g,b,a))
        """
        ticks = []
        for t in self.ticks:
            c = t.color
            ticks.append((self.ticks[t], c.getRgb()))
        state = {'mode': self.colorMode, 'ticks': ticks, 'ticksVisible': next(iter(self.ticks)).isVisible()}
        return state

    def restoreState(self, state):
        """
        Restore the gradient specified in state.
        
        ==============  ====================================================================
        **Arguments:**
        state           A dictionary with same structure as those returned by
                        :func:`saveState <pyqtgraph.GradientEditorItem.saveState>`
                      
                        Keys must include:
                      
                            - 'mode': hsv or rgb
                            - 'ticks': a list of tuples (pos, (r,g,b,a))
        ==============  ====================================================================
        """
        signalsBlocked = self.blockSignals(True)
        self.setColorMode(state['mode'])
        for t in list(self.ticks.keys()):
            self.removeTick(t, finish=False)
        for t in state['ticks']:
            c = QtGui.QColor(*t[1])
            self.addTick(t[0], c, finish=False)
        self.showTicks(state.get('ticksVisible', next(iter(self.ticks)).isVisible()))
        self.blockSignals(signalsBlocked)
        self.sigTicksChanged.emit(self)
        self.sigGradientChangeFinished.emit(self)

    def setColorMap(self, cm):
        signalsBlocked = self.blockSignals(True)
        self.setColorMode('rgb')
        for t in list(self.ticks.keys()):
            self.removeTick(t, finish=False)
        colors = cm.getColors(mode='qcolor')
        for i in range(len(cm.pos)):
            x = cm.pos[i]
            c = colors[i]
            self.addTick(x, c, finish=False)
        self.blockSignals(signalsBlocked)
        self.sigTicksChanged.emit(self)
        self.sigGradientChangeFinished.emit(self)

    def linkGradient(self, slaveGradient, connect=True):
        if connect:
            fn = lambda g, slave=slaveGradient: slave.restoreState(g.saveState())
            self.linkedGradients[id(slaveGradient)] = fn
            self.sigGradientChanged.connect(fn)
            self.sigGradientChanged.emit(self)
        else:
            fn = self.linkedGradients.get(id(slaveGradient), None)
            if fn:
                self.sigGradientChanged.disconnect(fn)