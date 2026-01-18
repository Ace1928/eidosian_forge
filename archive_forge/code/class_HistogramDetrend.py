import numpy as np
from ... import Point, PolyLineROI
from ... import functions as pgfn
from ... import metaarray as metaarray
from . import functions
from .common import CtrlNode, PlottingCtrlNode, metaArrayWrapper
class HistogramDetrend(CtrlNode):
    """Removes baseline from data by computing mode (from histogram) of beginning and end of data."""
    nodeName = 'HistogramDetrend'
    uiTemplate = [('windowSize', 'intSpin', {'value': 500, 'min': 10, 'max': 1000000, 'suffix': 'pts'}), ('numBins', 'intSpin', {'value': 50, 'min': 3, 'max': 1000000}), ('offsetOnly', 'check', {'checked': False})]

    def processData(self, data):
        s = self.stateGroup.state()
        return functions.histogramDetrend(data, window=s['windowSize'], bins=s['numBins'], offsetOnly=s['offsetOnly'])