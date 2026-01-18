import importlib
import os
from collections import OrderedDict
from numpy import ndarray
from .. import DataTreeWidget, FileDialog
from .. import configfile as configfile
from .. import dockarea as dockarea
from .. import functions as fn
from ..debug import printExc
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Qt import QtCore, QtWidgets
from . import FlowchartCtrlTemplate_generic as FlowchartCtrlTemplate
from . import FlowchartGraphicsView
from .library import LIBRARY
from .Node import Node
from .Terminal import Terminal
def addTerminal(self, name, **opts):
    term = Node.addTerminal(self, name, **opts)
    name = term.name()
    if opts['io'] == 'in':
        opts['io'] = 'out'
        opts['multi'] = False
        self.inputNode.sigTerminalAdded.disconnect(self.internalTerminalAdded)
        try:
            self.inputNode.addTerminal(name, **opts)
        finally:
            self.inputNode.sigTerminalAdded.connect(self.internalTerminalAdded)
    else:
        opts['io'] = 'in'
        self.outputNode.sigTerminalAdded.disconnect(self.internalTerminalAdded)
        try:
            self.outputNode.addTerminal(name, **opts)
        finally:
            self.outputNode.sigTerminalAdded.connect(self.internalTerminalAdded)
    return term