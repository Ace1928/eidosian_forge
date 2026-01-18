import numpy as np
from ...graphicsItems.LinearRegionItem import LinearRegionItem
from ...Qt import QtCore, QtWidgets
from ...widgets.TreeWidget import TreeWidget
from ..Node import Node
from . import functions
from .common import CtrlNode
class ColumnJoinNode(Node):
    """Concatenates record arrays and/or adds new columns"""
    nodeName = 'ColumnJoin'

    def __init__(self, name):
        Node.__init__(self, name, terminals={'output': {'io': 'out'}})
        self.ui = QtWidgets.QWidget()
        self.layout = QtWidgets.QGridLayout()
        self.ui.setLayout(self.layout)
        self.tree = TreeWidget()
        self.addInBtn = QtWidgets.QPushButton('+ Input')
        self.remInBtn = QtWidgets.QPushButton('- Input')
        self.layout.addWidget(self.tree, 0, 0, 1, 2)
        self.layout.addWidget(self.addInBtn, 1, 0)
        self.layout.addWidget(self.remInBtn, 1, 1)
        self.addInBtn.clicked.connect(self.addInput)
        self.remInBtn.clicked.connect(self.remInput)
        self.tree.sigItemMoved.connect(self.update)

    def ctrlWidget(self):
        return self.ui

    def addInput(self):
        term = Node.addInput(self, 'input', renamable=True, removable=True, multiable=True)
        item = QtWidgets.QTreeWidgetItem([term.name()])
        item.term = term
        term.joinItem = item
        self.tree.addTopLevelItem(item)

    def remInput(self):
        sel = self.tree.currentItem()
        term = sel.term
        term.joinItem = None
        sel.term = None
        self.tree.removeTopLevelItem(sel)
        self.removeTerminal(term)
        self.update()

    def process(self, display=True, **args):
        order = self.order()
        vals = []
        for name in order:
            if name not in args:
                continue
            val = args[name]
            if isinstance(val, np.ndarray) and len(val.dtype) > 0:
                vals.append(val)
            else:
                vals.append((name, None, val))
        return {'output': functions.concatenateColumns(vals)}

    def order(self):
        return [str(self.tree.topLevelItem(i).text(0)) for i in range(self.tree.topLevelItemCount())]

    def saveState(self):
        state = Node.saveState(self)
        state['order'] = self.order()
        return state

    def restoreState(self, state):
        Node.restoreState(self, state)
        inputs = self.inputs()
        for name in [n for n in state['order'] if n not in inputs]:
            Node.addInput(self, name, renamable=True, removable=True, multiable=True)
        inputs = self.inputs()
        order = [name for name in state['order'] if name in inputs]
        for name in inputs:
            if name not in order:
                order.append(name)
        self.tree.clear()
        for name in order:
            term = self[name]
            item = QtWidgets.QTreeWidgetItem([name])
            item.term = term
            term.joinItem = item
            self.tree.addTopLevelItem(item)

    def terminalRenamed(self, term, oldName):
        Node.terminalRenamed(self, term, oldName)
        item = term.joinItem
        item.setText(0, term.name())
        self.update()