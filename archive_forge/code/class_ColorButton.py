from ast import literal_eval
import copy
import datetime
import logging
from numbers import Integral, Real
from matplotlib import _api, colors as mcolors
from matplotlib.backends.qt_compat import _to_int, QtGui, QtWidgets, QtCore
class ColorButton(QtWidgets.QPushButton):
    """
    Color choosing push button
    """
    colorChanged = QtCore.Signal(QtGui.QColor)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(20, 20)
        self.setIconSize(QtCore.QSize(12, 12))
        self.clicked.connect(self.choose_color)
        self._color = QtGui.QColor()

    def choose_color(self):
        color = QtWidgets.QColorDialog.getColor(self._color, self.parentWidget(), '', QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if color.isValid():
            self.set_color(color)

    def get_color(self):
        return self._color

    @QtCore.Slot(QtGui.QColor)
    def set_color(self, color):
        if color != self._color:
            self._color = color
            self.colorChanged.emit(self._color)
            pixmap = QtGui.QPixmap(self.iconSize())
            pixmap.fill(color)
            self.setIcon(QtGui.QIcon(pixmap))
    color = QtCore.Property(QtGui.QColor, get_color, set_color)