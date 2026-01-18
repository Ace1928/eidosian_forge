import numpy as np
from ...graphicsItems.LinearRegionItem import LinearRegionItem
from ...Qt import QtCore, QtWidgets
from ...widgets.TreeWidget import TreeWidget
from ..Node import Node
from . import functions
from .common import CtrlNode
class Stdev(CtrlNode):
    """Calculate the standard deviation of an array across an axis.
    """
    nodeName = 'Stdev'
    uiTemplate = [('axis', 'intSpin', {'value': -0, 'min': -1, 'max': 1000000})]

    def processData(self, data):
        s = self.stateGroup.state()
        ax = None if s['axis'] == -1 else s['axis']
        return data.std(axis=ax)