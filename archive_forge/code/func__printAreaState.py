import weakref
from ..Qt import QtWidgets
from .Container import Container, HContainer, TContainer, VContainer
from .Dock import Dock
from .DockDrop import DockDrop
def _printAreaState(self, area, indent=0):
    if area[0] == 'dock':
        print('  ' * indent + area[0] + ' ' + str(area[1:]))
        return
    else:
        print('  ' * indent + area[0])
        for ch in area[1]:
            self._printAreaState(ch, indent + 1)