from .. import functions as fn
from ..Qt import QtWidgets
from .GraphicsWidget import GraphicsWidget
from .LabelItem import LabelItem
from .PlotItem import PlotItem
from .ViewBox import ViewBox
def _updateItemBorder(self):
    if self.border is None:
        return
    item = self.sender()
    if item is None:
        return
    r = item.mapRectToParent(item.boundingRect())
    self.itemBorders[item].setRect(r)