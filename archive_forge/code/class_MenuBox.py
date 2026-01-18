import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
class MenuBox(pg.GraphicsObject):
    """
    This class draws a rectangular area. Right-clicking inside the area will
    raise a custom context menu which also includes the context menus of
    its parents.    
    """

    def __init__(self, name):
        self.name = name
        self.pen = pg.mkPen('r')
        self.menu = None
        pg.GraphicsObject.__init__(self)

    def boundingRect(self):
        return QtCore.QRectF(0, 0, 10, 10)

    def paint(self, p, *args):
        p.setPen(self.pen)
        p.drawRect(self.boundingRect())

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            if self.raiseContextMenu(ev):
                ev.accept()

    def raiseContextMenu(self, ev):
        menu = self.getContextMenus()
        menu = self.scene().addParentContextMenus(self, menu, ev)
        pos = ev.screenPos()
        menu.popup(QtCore.QPoint(int(pos.x()), int(pos.y())))
        return True

    def getContextMenus(self, event=None):
        if self.menu is None:
            self.menu = QtWidgets.QMenu()
            self.menu.setTitle(self.name + ' options..')
            green = QtGui.QAction('Turn green', self.menu)
            green.triggered.connect(self.setGreen)
            self.menu.addAction(green)
            self.menu.green = green
            blue = QtGui.QAction('Turn blue', self.menu)
            blue.triggered.connect(self.setBlue)
            self.menu.addAction(blue)
            self.menu.green = blue
            alpha = QtWidgets.QWidgetAction(self.menu)
            alphaSlider = QtWidgets.QSlider()
            alphaSlider.setOrientation(QtCore.Qt.Orientation.Horizontal)
            alphaSlider.setMaximum(255)
            alphaSlider.setValue(255)
            alphaSlider.valueChanged.connect(self.setAlpha)
            alpha.setDefaultWidget(alphaSlider)
            self.menu.addAction(alpha)
            self.menu.alpha = alpha
            self.menu.alphaSlider = alphaSlider
        return self.menu

    def setGreen(self):
        self.pen = pg.mkPen('g')
        self.update()

    def setBlue(self):
        self.pen = pg.mkPen('b')
        self.update()

    def setAlpha(self, a):
        self.setOpacity(a / 255.0)