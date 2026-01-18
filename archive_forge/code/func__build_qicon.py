import os.path as op
import warnings
from ..Qt import QtGui, QtWidgets
def _build_qicon(self):
    icon = QtGui.QIcon(op.join(op.dirname(__file__), self._path))
    name = self._path.split('.')[0]
    _ICON_REGISTRY[name] = icon
    self._icon = icon