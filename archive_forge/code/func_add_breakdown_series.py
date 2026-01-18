import sys
from PySide2.QtCore import Qt
from PySide2.QtGui import QColor, QFont, QPainter
from PySide2.QtWidgets import QApplication, QMainWindow
from PySide2.QtCharts import QtCharts
def add_breakdown_series(self, breakdown_series, color):
    font = QFont('Arial', 8)
    main_slice = MainSlice(breakdown_series)
    main_slice.setName(breakdown_series.name())
    main_slice.setValue(breakdown_series.sum())
    self.main_series.append(main_slice)
    main_slice.setBrush(color)
    main_slice.setLabelVisible()
    main_slice.setLabelColor(Qt.white)
    main_slice.setLabelPosition(QtCharts.QPieSlice.LabelInsideHorizontal)
    main_slice.setLabelFont(font)
    breakdown_series.setPieSize(0.8)
    breakdown_series.setHoleSize(0.7)
    breakdown_series.setLabelsVisible()
    for pie_slice in breakdown_series.slices():
        color = QColor(color).lighter(115)
        pie_slice.setBrush(color)
        pie_slice.setLabelFont(font)
    self.addSeries(breakdown_series)
    self.recalculate_angles()
    self.update_legend_markers()