import numpy as np
from ... import Point, PolyLineROI
from ... import functions as pgfn
from ... import metaarray as metaarray
from . import functions
from .common import CtrlNode, PlottingCtrlNode, metaArrayWrapper
class Subsample(CtrlNode):
    """Downsample by selecting every Nth sample."""
    nodeName = 'Subsample'
    uiTemplate = [('n', 'intSpin', {'min': 1, 'max': 1000000})]

    def processData(self, data):
        return data[::self.ctrls['n'].value()]