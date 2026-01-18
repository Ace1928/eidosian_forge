from ast import literal_eval
import copy
import datetime
import logging
from numbers import Integral, Real
from matplotlib import _api, colors as mcolors
from matplotlib.backends.qt_compat import _to_int, QtGui, QtWidgets, QtCore
class FormTabWidget(QtWidgets.QWidget):
    update_buttons = QtCore.Signal()

    def __init__(self, datalist, comment='', parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout()
        self.tabwidget = QtWidgets.QTabWidget()
        layout.addWidget(self.tabwidget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.widgetlist = []
        for data, title, comment in datalist:
            if len(data[0]) == 3:
                widget = FormComboWidget(data, comment=comment, parent=self)
            else:
                widget = FormWidget(data, with_margin=True, comment=comment, parent=self)
            index = self.tabwidget.addTab(widget, title)
            self.tabwidget.setTabToolTip(index, comment)
            self.widgetlist.append(widget)

    def setup(self):
        for widget in self.widgetlist:
            widget.setup()

    def get(self):
        return [widget.get() for widget in self.widgetlist]