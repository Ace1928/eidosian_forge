import numpy as np
from ... import Point, PolyLineROI
from ... import functions as pgfn
from ... import metaarray as metaarray
from . import functions
from .common import CtrlNode, PlottingCtrlNode, metaArrayWrapper
class Bessel(CtrlNode):
    """Bessel filter. Input data must have time values."""
    nodeName = 'BesselFilter'
    uiTemplate = [('band', 'combo', {'values': ['lowpass', 'highpass'], 'index': 0}), ('cutoff', 'spin', {'value': 1000.0, 'step': 1, 'dec': True, 'bounds': [0.0, None], 'suffix': 'Hz', 'siPrefix': True}), ('order', 'intSpin', {'value': 4, 'min': 1, 'max': 16}), ('bidir', 'check', {'checked': True})]

    def processData(self, data):
        s = self.stateGroup.state()
        if s['band'] == 'lowpass':
            mode = 'low'
        else:
            mode = 'high'
        return functions.besselFilter(data, bidir=s['bidir'], btype=mode, cutoff=s['cutoff'], order=s['order'])