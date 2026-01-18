import collections.abc
import os
import warnings
import weakref
import numpy as np
from ... import functions as fn
from ... import icons
from ...Qt import QtCore, QtWidgets
from ...WidgetGroup import WidgetGroup
from ...widgets.FileDialog import FileDialog
from ..AxisItem import AxisItem
from ..ButtonItem import ButtonItem
from ..GraphicsWidget import GraphicsWidget
from ..InfiniteLine import InfiniteLine
from ..LabelItem import LabelItem
from ..LegendItem import LegendItem
from ..PlotCurveItem import PlotCurveItem
from ..PlotDataItem import PlotDataItem
from ..ScatterPlotItem import ScatterPlotItem
from ..ViewBox import ViewBox
from . import plotConfigTemplate_generic as ui_template
def addAvgCurve(self, curve):
    remKeys = []
    addKeys = []
    if self.ctrl.avgParamList.count() > 0:
        for i in range(self.ctrl.avgParamList.count()):
            item = self.ctrl.avgParamList.item(i)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                remKeys.append(str(item.text()))
            else:
                addKeys.append(str(item.text()))
        if len(remKeys) < 1:
            return
    p = self.itemMeta.get(curve, {}).copy()
    for k in p:
        if type(k) is tuple:
            p['.'.join(k)] = p[k]
            del p[k]
    for rk in remKeys:
        if rk in p:
            del p[rk]
    for ak in addKeys:
        if ak not in p:
            p[ak] = None
    key = tuple(p.items())
    if key not in self.avgCurves:
        plot = PlotDataItem()
        plot.setPen(self.avgPen)
        plot.setShadowPen(self.avgShadowPen)
        plot.setAlpha(1.0, False)
        plot.setZValue(100)
        self.addItem(plot, skipAverage=True)
        self.avgCurves[key] = [0, plot]
    self.avgCurves[key][0] += 1
    n, plot = self.avgCurves[key]
    x, y = curve.getData()
    stepMode = curve.opts['stepMode']
    if plot.yData is not None and y.shape == plot.yData.shape:
        newData = plot.yData * (n - 1) / float(n) + y * 1.0 / float(n)
        plot.setData(plot.xData, newData, stepMode=stepMode)
    else:
        plot.setData(x, y, stepMode=stepMode)