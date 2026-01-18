import numpy as np
from .. import Qt, colormap
from .. import functions as fn
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
def _rerender(self, *, autoLevels):
    self.qpicture = None
    if self.z is not None:
        if self.levels is None or autoLevels:
            z_min = self.z.min()
            z_max = self.z.max()
            self.setLevels((z_min, z_max), update=False)
        self.qpicture = self._drawPicture()