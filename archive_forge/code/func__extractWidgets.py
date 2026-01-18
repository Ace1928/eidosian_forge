from time import perf_counter
from ..Qt import QtCore, QtGui, QtWidgets
def _extractWidgets(self):
    if self._nestableWidgets is None:
        label = [ch for ch in self.children() if isinstance(ch, QtWidgets.QLabel)][0]
        bar = [ch for ch in self.children() if isinstance(ch, QtWidgets.QProgressBar)][0]
        btn = [ch for ch in self.children() if isinstance(ch, QtWidgets.QPushButton)][0]
        sw = ProgressWidget(label, bar)
        self._nestableWidgets = (sw, btn)
    return self._nestableWidgets