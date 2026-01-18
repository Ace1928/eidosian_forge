from time import perf_counter
from ..Qt import QtCore, QtGui, QtWidgets
def _prepareNesting(self):
    if self._nestingReady is False:
        self._topLayout = QtWidgets.QGridLayout()
        self.setLayout(self._topLayout)
        self._topLayout.setContentsMargins(0, 0, 0, 0)
        self.nestedVBox = QtWidgets.QWidget()
        self._topLayout.addWidget(self.nestedVBox, 0, 0, 1, 2)
        self.nestedLayout = QtWidgets.QVBoxLayout()
        self.nestedVBox.setLayout(self.nestedLayout)
        bar, btn = self._extractWidgets()
        self.nestedLayout.addWidget(bar)
        self._subBars.append(bar)
        self._topLayout.addWidget(btn, 1, 1, 1, 1)
        self._topLayout.setColumnStretch(0, 100)
        self._topLayout.setColumnStretch(1, 1)
        self._topLayout.setRowStretch(0, 100)
        self._topLayout.setRowStretch(1, 1)
        self._nestingReady = True