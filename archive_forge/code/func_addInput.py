import numpy as np
from ...graphicsItems.LinearRegionItem import LinearRegionItem
from ...Qt import QtCore, QtWidgets
from ...widgets.TreeWidget import TreeWidget
from ..Node import Node
from . import functions
from .common import CtrlNode
def addInput(self):
    term = Node.addInput(self, 'input', renamable=True, removable=True, multiable=True)
    item = QtWidgets.QTreeWidgetItem([term.name()])
    item.term = term
    term.joinItem = item
    self.tree.addTopLevelItem(item)