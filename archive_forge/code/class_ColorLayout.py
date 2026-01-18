from ast import literal_eval
import copy
import datetime
import logging
from numbers import Integral, Real
from matplotlib import _api, colors as mcolors
from matplotlib.backends.qt_compat import _to_int, QtGui, QtWidgets, QtCore
class ColorLayout(QtWidgets.QHBoxLayout):
    """Color-specialized QLineEdit layout"""

    def __init__(self, color, parent=None):
        super().__init__()
        assert isinstance(color, QtGui.QColor)
        self.lineedit = QtWidgets.QLineEdit(mcolors.to_hex(color.getRgbF(), keep_alpha=True), parent)
        self.lineedit.editingFinished.connect(self.update_color)
        self.addWidget(self.lineedit)
        self.colorbtn = ColorButton(parent)
        self.colorbtn.color = color
        self.colorbtn.colorChanged.connect(self.update_text)
        self.addWidget(self.colorbtn)

    def update_color(self):
        color = self.text()
        qcolor = to_qcolor(color)
        self.colorbtn.color = qcolor

    def update_text(self, color):
        self.lineedit.setText(mcolors.to_hex(color.getRgbF(), keep_alpha=True))

    def text(self):
        return self.lineedit.text()