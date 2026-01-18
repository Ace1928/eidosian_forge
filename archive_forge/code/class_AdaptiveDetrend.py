import numpy as np
from ... import Point, PolyLineROI
from ... import functions as pgfn
from ... import metaarray as metaarray
from . import functions
from .common import CtrlNode, PlottingCtrlNode, metaArrayWrapper
class AdaptiveDetrend(CtrlNode):
    """Removes baseline from data, ignoring anomalous events"""
    nodeName = 'AdaptiveDetrend'
    uiTemplate = [('threshold', 'doubleSpin', {'value': 3.0, 'min': 0, 'max': 1000000})]

    def processData(self, data):
        return functions.adaptiveDetrend(data, threshold=self.ctrls['threshold'].value())