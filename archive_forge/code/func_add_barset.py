import sys
from PySide2.QtCore import Qt, QRectF
from PySide2.QtGui import QBrush, QColor, QPainter, QPen
from PySide2.QtWidgets import (QApplication, QDoubleSpinBox,
from PySide2.QtCharts import QtCharts
def add_barset(self):
    series_count = self.series.count()
    bar_set = QtCharts.QBarSet('set {}'.format(series_count))
    delta = series_count * 0.1
    bar_set.append([1 + delta, 2 + delta, 3 + delta, 4 + delta])
    self.series.append(bar_set)