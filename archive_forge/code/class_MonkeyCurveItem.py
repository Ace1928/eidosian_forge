import argparse
import itertools
import numpy as np
from utils import FrameCounter
import pyqtgraph as pg
import pyqtgraph.functions as fn
import pyqtgraph.parametertree as ptree
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
class MonkeyCurveItem(pg.PlotCurveItem):

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.monkey_mode = ''

    def setMethod(self, value):
        self.monkey_mode = value

    def paint(self, painter, opt, widget):
        if self.monkey_mode not in ['drawPolyline']:
            return super().paint(painter, opt, widget)
        painter.setRenderHint(painter.RenderHint.Antialiasing, self.opts['antialias'])
        painter.setPen(pg.mkPen(self.opts['pen']))
        if self.monkey_mode == 'drawPolyline':
            painter.drawPolyline(fn.arrayToQPolygonF(self.xData, self.yData))