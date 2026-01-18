import sys
from collections import OrderedDict
from .. import functions as fn
from ..debug import printExc
from ..graphicsItems.GraphicsObject import GraphicsObject
from ..Qt import QtCore, QtGui, QtWidgets
from .Terminal import Terminal
def addOutputFromMenu(self):
    self.node.addOutput(renamable=True, removable=True, multiable=False)