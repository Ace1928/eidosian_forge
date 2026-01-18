import numpy as np
from ...graphicsItems.LinearRegionItem import LinearRegionItem
from ...Qt import QtCore, QtWidgets
from ...widgets.TreeWidget import TreeWidget
from ..Node import Node
from . import functions
from .common import CtrlNode
class ColumnSelectNode(Node):
    """Select named columns from a record array or MetaArray."""
    nodeName = 'ColumnSelect'

    def __init__(self, name):
        Node.__init__(self, name, terminals={'In': {'io': 'in'}})
        self.columns = set()
        self.columnList = QtWidgets.QListWidget()
        self.axis = 0
        self.columnList.itemChanged.connect(self.itemChanged)

    def process(self, In, display=True):
        if display:
            self.updateList(In)
        out = {}
        if hasattr(In, 'implements') and In.implements('MetaArray'):
            for c in self.columns:
                out[c] = In[self.axis:c]
        elif isinstance(In, np.ndarray) and In.dtype.fields is not None:
            for c in self.columns:
                out[c] = In[c]
        else:
            self.In.setValueAcceptable(False)
            raise Exception('Input must be MetaArray or ndarray with named fields')
        return out

    def ctrlWidget(self):
        return self.columnList

    def updateList(self, data):
        if hasattr(data, 'implements') and data.implements('MetaArray'):
            cols = data.listColumns()
            for ax in cols:
                if len(cols[ax]) > 0:
                    self.axis = ax
                    cols = set(cols[ax])
                    break
        else:
            cols = list(data.dtype.fields.keys())
        rem = set()
        for c in self.columns:
            if c not in cols:
                self.removeTerminal(c)
                rem.add(c)
        self.columns -= rem
        self.columnList.blockSignals(True)
        self.columnList.clear()
        for c in cols:
            item = QtWidgets.QListWidgetItem(c)
            item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            if c in self.columns:
                item.setCheckState(QtCore.Qt.CheckState.Checked)
            else:
                item.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.columnList.addItem(item)
        self.columnList.blockSignals(False)

    def itemChanged(self, item):
        col = str(item.text())
        if item.checkState() == QtCore.Qt.CheckState.Checked:
            if col not in self.columns:
                self.columns.add(col)
                self.addOutput(col)
        elif col in self.columns:
            self.columns.remove(col)
            self.removeTerminal(col)
        self.update()

    def saveState(self):
        state = Node.saveState(self)
        state['columns'] = list(self.columns)
        return state

    def restoreState(self, state):
        Node.restoreState(self, state)
        self.columns = set(state.get('columns', []))
        for c in self.columns:
            self.addOutput(c)