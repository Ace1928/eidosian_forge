import sys
from PySide2.QtCore import Qt
from PySide2.QtGui import QColor, QFont, QPainter
from PySide2.QtWidgets import QApplication, QMainWindow
from PySide2.QtCharts import QtCharts
class DonutBreakdownChart(QtCharts.QChart):

    def __init__(self, parent=None):
        super(DonutBreakdownChart, self).__init__(QtCharts.QChart.ChartTypeCartesian, parent, Qt.WindowFlags())
        self.main_series = QtCharts.QPieSeries()
        self.main_series.setPieSize(0.7)
        self.addSeries(self.main_series)

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

    def recalculate_angles(self):
        angle = 0
        slices = self.main_series.slices()
        for pie_slice in slices:
            breakdown_series = pie_slice.get_breakdown_series()
            breakdown_series.setPieStartAngle(angle)
            angle += pie_slice.percentage() * 360.0
            breakdown_series.setPieEndAngle(angle)

    def update_legend_markers(self):
        for series in self.series():
            markers = self.legend().markers(series)
            for marker in markers:
                if series == self.main_series:
                    marker.setVisible(False)
                else:
                    marker.setLabel('{} {:.2f}%'.format(marker.slice().label(), marker.slice().percentage() * 100, 0))
                    marker.setFont(QFont('Arial', 8))