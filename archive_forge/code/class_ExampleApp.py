import os
import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, loadUiType
class ExampleApp(QtWidgets.QMainWindow, Design):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        now = time.time()
        timestamps = np.linspace(now - 6 * 30 * 24 * 3600, now, 100)
        self.curve = self.plotWidget.plot(x=timestamps, y=np.random.rand(100), symbol='o', symbolSize=5, pen=BLUE)
        self.plotWidget.setAxisItems({'bottom': pg.DateAxisItem()})
        self.plotWidget.showGrid(x=True, y=True)