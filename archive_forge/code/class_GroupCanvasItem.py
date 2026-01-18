import importlib
from .. import ItemGroup, SRTTransform, debug
from .. import functions as fn
from ..graphicsItems.ROI import ROI
from ..Qt import QT_LIB, QtCore, QtWidgets
from . import TransformGuiTemplate_generic as ui_template
class GroupCanvasItem(CanvasItem):
    """
    Canvas item used for grouping others
    """

    def __init__(self, **opts):
        defOpts = {'movable': False, 'scalable': False}
        defOpts.update(opts)
        item = ItemGroup()
        CanvasItem.__init__(self, item, **defOpts)