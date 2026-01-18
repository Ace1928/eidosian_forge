import numpy as np
import pyqtgraph as pg
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.flowchart import Flowchart, Node
from pyqtgraph.flowchart.library.common import CtrlNode
from pyqtgraph.Qt import QtWidgets
class UnsharpMaskNode(CtrlNode):
    """Return the input data passed through an unsharp mask."""
    nodeName = 'UnsharpMask'
    uiTemplate = [('sigma', 'spin', {'value': 1.0, 'step': 1.0, 'bounds': [0.0, None]}), ('strength', 'spin', {'value': 1.0, 'dec': True, 'step': 0.5, 'minStep': 0.01, 'bounds': [0.0, None]})]

    def __init__(self, name):
        terminals = {'dataIn': dict(io='in'), 'dataOut': dict(io='out')}
        CtrlNode.__init__(self, name, terminals=terminals)

    def process(self, dataIn, display=True):
        sigma = self.ctrls['sigma'].value()
        strength = self.ctrls['strength'].value()
        output = dataIn - strength * pg.gaussianFilter(dataIn, (sigma, sigma))
        return {'dataOut': output}