from copy import copy
from ase.gui.i18n import _
import numpy as np
import ase
import ase.data
import ase.gui.ui as ui
from ase.cluster.cubic import FaceCenteredCubic, BodyCenteredCubic, SimpleCubic
from ase.cluster.hexagonal import HexagonalClosedPacked, Graphite
from ase.cluster import wulff_construction
from ase.gui.widgets import Element, pybutton
import ase
import ase
from ase.cluster import wulff_construction
def add_direction(self, direction, layers, energy):
    i = len(self.direction_table_rows)
    if self.method.value == 'wulff':
        spin = ui.SpinBox(energy, 0.0, 1000.0, 0.1, self.update)
    else:
        spin = ui.SpinBox(layers, 1, 100, 1, self.update)
    up = ui.Button(_('Up'), self.row_swap_next, i - 1)
    down = ui.Button(_('Down'), self.row_swap_next, i)
    delete = ui.Button(_('Delete'), self.row_delete, i)
    self.direction_table_rows.add([str(direction) + ':', spin, up, down, delete])
    up.active = i > 0
    down.active = False
    delete.active = i > 0
    if i > 0:
        down, delete = self.direction_table_rows[-2][3:]
        down.active = True
        delete.active = True