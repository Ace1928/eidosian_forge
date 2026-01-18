import numpy as np
from ... import Point, PolyLineROI
from ... import functions as pgfn
from ... import metaarray as metaarray
from . import functions
from .common import CtrlNode, PlottingCtrlNode, metaArrayWrapper
class Downsample(CtrlNode):
    """Downsample by averaging samples together."""
    nodeName = 'Downsample'
    uiTemplate = [('n', 'intSpin', {'min': 1, 'max': 1000000})]

    def processData(self, data):
        return functions.downsample(data, self.ctrls['n'].value(), axis=0)