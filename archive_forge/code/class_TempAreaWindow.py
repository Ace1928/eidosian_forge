import weakref
from ..Qt import QtWidgets
from .Container import Container, HContainer, TContainer, VContainer
from .Dock import Dock
from .DockDrop import DockDrop
class TempAreaWindow(QtWidgets.QWidget):

    def __init__(self, area, **kwargs):
        QtWidgets.QWidget.__init__(self, **kwargs)
        self.layout = QtWidgets.QGridLayout()
        self.setLayout(self.layout)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.dockarea = area
        self.layout.addWidget(area)

    def closeEvent(self, *args):
        docks = self.dockarea.findAll()[1]
        for dock in docks.values():
            if hasattr(dock, 'orig_area'):
                dock.orig_area.addDock(dock)
        self.dockarea.clear()
        super().closeEvent(*args)