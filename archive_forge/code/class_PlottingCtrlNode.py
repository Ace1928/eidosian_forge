import numpy as np
from ...Qt import QtCore, QtWidgets
from ...WidgetGroup import WidgetGroup
from ...widgets.ColorButton import ColorButton
from ...widgets.SpinBox import SpinBox
from ..Node import Node
class PlottingCtrlNode(CtrlNode):
    """Abstract class for CtrlNodes that can connect to plots."""

    def __init__(self, name, ui=None, terminals=None):
        CtrlNode.__init__(self, name, ui=ui, terminals=terminals)
        self.plotTerminal = self.addOutput('plot', optional=True)

    def connected(self, term, remote):
        CtrlNode.connected(self, term, remote)
        if term is not self.plotTerminal:
            return
        node = remote.node()
        node.sigPlotChanged.connect(self.connectToPlot)
        self.connectToPlot(node)

    def disconnected(self, term, remote):
        CtrlNode.disconnected(self, term, remote)
        if term is not self.plotTerminal:
            return
        remote.node().sigPlotChanged.disconnect(self.connectToPlot)
        self.disconnectFromPlot(remote.node().getPlot())

    def connectToPlot(self, node):
        """Define what happens when the node is connected to a plot"""
        raise Exception('Must be re-implemented in subclass')

    def disconnectFromPlot(self, plot):
        """Define what happens when the node is disconnected from a plot"""
        raise Exception('Must be re-implemented in subclass')

    def process(self, In, display=True):
        out = CtrlNode.process(self, In, display)
        out['plot'] = None
        return out