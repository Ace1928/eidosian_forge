import numpy as np
from ... import ComboBox, PlotDataItem
from ...graphicsItems.ScatterPlotItem import ScatterPlotItem
from ...Qt import QtCore, QtGui, QtWidgets
from ..Node import Node
from .common import CtrlNode
class CanvasNode(Node):
    """Connection to a Canvas widget."""
    nodeName = 'CanvasWidget'

    def __init__(self, name):
        Node.__init__(self, name, terminals={'In': {'io': 'in', 'multi': True}})
        self.canvas = None
        self.items = {}

    def disconnected(self, localTerm, remoteTerm):
        if localTerm is self.In and remoteTerm in self.items:
            self.canvas.removeItem(self.items[remoteTerm])
            del self.items[remoteTerm]

    def setCanvas(self, canvas):
        self.canvas = canvas

    def getCanvas(self):
        return self.canvas

    def process(self, In, display=True):
        if display:
            items = set()
            for name, vals in In.items():
                if vals is None:
                    continue
                if type(vals) is not list:
                    vals = [vals]
                for val in vals:
                    vid = id(val)
                    if vid in self.items:
                        items.add(vid)
                    else:
                        self.canvas.addItem(val)
                        item = val
                        self.items[vid] = item
                        items.add(vid)
            for vid in list(self.items.keys()):
                if vid not in items:
                    self.canvas.removeItem(self.items[vid])
                    del self.items[vid]