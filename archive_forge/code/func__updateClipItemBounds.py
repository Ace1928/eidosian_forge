from .. import debug
from .. import functions as fn
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
from .InfiniteLine import InfiniteLine
def _updateClipItemBounds(self):
    item_vb = self.clipItem.getViewBox()
    if item_vb is None:
        return
    item_bounds = item_vb.childrenBounds(items=(self.clipItem,))
    if item_bounds == self._clipItemBoundsCache or None in item_bounds:
        return
    self._clipItemBoundsCache = item_bounds
    if self.orientation in ('horizontal', LinearRegionItem.Horizontal):
        self._setBounds(item_bounds[1])
    else:
        self._setBounds(item_bounds[0])