import numpy as np
import pyqtgraph as pg
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.flowchart import Flowchart, Node
from pyqtgraph.flowchart.library.common import CtrlNode
from pyqtgraph.Qt import QtWidgets
class ImageViewNode(Node):
    """Node that displays image data in an ImageView widget"""
    nodeName = 'ImageView'

    def __init__(self, name):
        self.view = None
        Node.__init__(self, name, terminals={'data': {'io': 'in'}})

    def setView(self, view):
        self.view = view

    def process(self, data, display=True):
        if display and self.view is not None:
            if data is None:
                self.view.setImage(np.zeros((1, 1)))
            else:
                self.view.setImage(data)