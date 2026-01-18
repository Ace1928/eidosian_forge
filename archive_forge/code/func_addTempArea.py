import weakref
from ..Qt import QtWidgets
from .Container import Container, HContainer, TContainer, VContainer
from .Dock import Dock
from .DockDrop import DockDrop
def addTempArea(self):
    if self.home is None:
        area = DockArea(temporary=True, home=self)
        self.tempAreas.append(area)
        win = TempAreaWindow(area)
        area.win = win
        win.show()
    else:
        area = self.home.addTempArea()
    return area