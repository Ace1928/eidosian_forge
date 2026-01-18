import sys
from PySide2.QtCore import Qt
from PySide2.QtGui import QColor, QFont, QPainter
from PySide2.QtWidgets import QApplication, QMainWindow
from PySide2.QtCharts import QtCharts
class MainSlice(QtCharts.QPieSlice):

    def __init__(self, breakdown_series, parent=None):
        super(MainSlice, self).__init__(parent)
        self.breakdown_series = breakdown_series
        self.name = None
        self.percentageChanged.connect(self.update_label)

    def get_breakdown_series(self):
        return self.breakdown_series

    def setName(self, name):
        self.name = name

    def name(self):
        return self.name

    def update_label(self):
        self.setLabel('{} {:.2f}%'.format(self.name, self.percentage() * 100))