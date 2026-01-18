import weakref
from ..Qt import QtCore, QtWidgets
from .Dock import Dock
class TContainer(Container, QtWidgets.QWidget):
    sigStretchChanged = QtCore.Signal()

    def __init__(self, area):
        QtWidgets.QWidget.__init__(self)
        Container.__init__(self, area)
        self.layout = QtWidgets.QGridLayout()
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)
        self.hTabLayout = QtWidgets.QHBoxLayout()
        self.hTabBox = QtWidgets.QWidget()
        self.hTabBox.setLayout(self.hTabLayout)
        self.hTabLayout.setSpacing(2)
        self.hTabLayout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.hTabBox, 0, 1)
        self.stack = StackedWidget(container=self)
        self.layout.addWidget(self.stack, 1, 1)
        self.setLayout(self.layout)
        for n in ['count', 'widget', 'indexOf']:
            setattr(self, n, getattr(self.stack, n))

    def _insertItem(self, item, index):
        if not isinstance(item, Dock):
            raise Exception('Tab containers may hold only docks, not other containers.')
        self.stack.insertWidget(index, item)
        self.hTabLayout.insertWidget(index, item.label)
        item.label.sigClicked.connect(self.tabClicked)
        self.tabClicked(item.label)

    def tabClicked(self, tab, ev=None):
        if ev is None or ev.button() == QtCore.Qt.MouseButton.LeftButton:
            for i in range(self.count()):
                w = self.widget(i)
                if w is tab.dock:
                    w.label.setDim(False)
                    self.stack.setCurrentIndex(i)
                else:
                    w.label.setDim(True)

    def raiseDock(self, dock):
        """Move *dock* to the top of the stack"""
        self.stack.currentWidget().label.setDim(True)
        self.stack.setCurrentWidget(dock)
        dock.label.setDim(False)

    def type(self):
        return 'tab'

    def saveState(self):
        return {'index': self.stack.currentIndex()}

    def restoreState(self, state):
        self.stack.setCurrentIndex(state['index'])

    def updateStretch(self):
        x = 0
        y = 0
        for i in range(self.count()):
            wx, wy = self.widget(i).stretch()
            x = max(x, wx)
            y = max(y, wy)
        self.setStretch(x, y)