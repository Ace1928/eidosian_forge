import functools
import os
import sys
import traceback
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (
import matplotlib.backends.qt_editor.figureoptions as figureoptions
from . import qt_compat
from .qt_compat import (
def _export_values(self):
    self._export_values_dialog = QtWidgets.QDialog()
    layout = QtWidgets.QVBoxLayout()
    self._export_values_dialog.setLayout(layout)
    text = QtWidgets.QPlainTextEdit()
    text.setReadOnly(True)
    layout.addWidget(text)
    text.setPlainText(',\n'.join((f'{attr}={spinbox.value():.3}' for attr, spinbox in self._spinboxes.items())))
    size = text.maximumSize()
    size.setHeight(QtGui.QFontMetrics(text.document().defaultFont()).size(0, text.toPlainText()).height() + 20)
    text.setMaximumSize(size)
    self._export_values_dialog.show()