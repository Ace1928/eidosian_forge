import numpy as np
from ...Qt import QtCore, QtWidgets
from ...WidgetGroup import WidgetGroup
from ...widgets.ColorButton import ColorButton
from ...widgets.SpinBox import SpinBox
from ..Node import Node
class CtrlNode(Node):
    """Abstract class for nodes with auto-generated control UI"""
    sigStateChanged = QtCore.Signal(object)

    def __init__(self, name, ui=None, terminals=None):
        if terminals is None:
            terminals = {'In': {'io': 'in'}, 'Out': {'io': 'out', 'bypass': 'In'}}
        Node.__init__(self, name=name, terminals=terminals)
        if ui is None:
            if hasattr(self, 'uiTemplate'):
                ui = self.uiTemplate
            else:
                ui = []
        self.ui, self.stateGroup, self.ctrls = generateUi(ui)
        self.stateGroup.sigChanged.connect(self.changed)

    def ctrlWidget(self):
        return self.ui

    def changed(self):
        self.update()
        self.sigStateChanged.emit(self)

    def process(self, In, display=True):
        out = self.processData(In)
        return {'Out': out}

    def saveState(self):
        state = Node.saveState(self)
        state['ctrl'] = self.stateGroup.state()
        return state

    def restoreState(self, state):
        Node.restoreState(self, state)
        if self.stateGroup is not None:
            self.stateGroup.setState(state.get('ctrl', {}))

    def hideRow(self, name):
        w = self.ctrls[name]
        l = self.ui.layout().labelForField(w)
        w.hide()
        l.hide()

    def showRow(self, name):
        w = self.ctrls[name]
        l = self.ui.layout().labelForField(w)
        w.show()
        l.show()