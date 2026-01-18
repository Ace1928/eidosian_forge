from ... import functions as fn
from ...widgets.ColorButton import ColorButton
from .basetypes import SimpleParameter, WidgetParameterItem
def _interpretValue(self, v):
    return fn.mkColor(v)