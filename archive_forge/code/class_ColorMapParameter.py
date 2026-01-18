from ...colormap import ColorMap
from ...Qt import QtCore
from ...widgets.GradientWidget import GradientWidget
from .basetypes import SimpleParameter, WidgetParameterItem
class ColorMapParameter(SimpleParameter):
    itemClass = ColorMapParameterItem

    def _interpretValue(self, v):
        if v is not None and (not isinstance(v, ColorMap)):
            raise TypeError('Cannot set colormap parameter from object %r' % v)
        return v