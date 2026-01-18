import numpy as np
from ... import Point, PolyLineROI
from ... import functions as pgfn
from ... import metaarray as metaarray
from . import functions
from .common import CtrlNode, PlottingCtrlNode, metaArrayWrapper
class RemovePeriodic(CtrlNode):
    nodeName = 'RemovePeriodic'
    uiTemplate = [('f0', 'spin', {'value': 60, 'suffix': 'Hz', 'siPrefix': True, 'min': 0, 'max': None}), ('harmonics', 'intSpin', {'value': 30, 'min': 0}), ('samples', 'intSpin', {'value': 1, 'min': 1})]

    def processData(self, data):
        times = data.xvals('Time')
        dt = times[1] - times[0]
        data1 = data.asarray()
        ft = np.fft.fft(data1)
        df = 1.0 / (len(data1) * dt)
        f0 = self.ctrls['f0'].value()
        for i in range(1, self.ctrls['harmonics'].value() + 2):
            f = f0 * i
            ind1 = int(np.floor(f / df))
            ind2 = int(np.ceil(f / df)) + (self.ctrls['samples'].value() - 1)
            if ind1 > len(ft) / 2.0:
                break
            mag = (abs(ft[ind1 - 1]) + abs(ft[ind2 + 1])) * 0.5
            for j in range(ind1, ind2 + 1):
                phase = np.angle(ft[j])
                re = mag * np.cos(phase)
                im = mag * np.sin(phase)
                ft[j] = re + im * 1j
                ft[len(ft) - j] = re - im * 1j
        data2 = np.fft.ifft(ft).real
        ma = metaarray.MetaArray(data2, info=data.infoCopy())
        return ma