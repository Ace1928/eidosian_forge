from ... import colormap
from ...widgets.ColorMapButton import ColorMapButton
from .basetypes import Parameter, WidgetParameterItem
class ColorMapLutParameterItem(WidgetParameterItem):

    def makeWidget(self):
        w = ColorMapButton()
        w.sigChanged = w.sigColorMapChanged
        w.value = w.colorMap
        w.setValue = w.setColorMap
        self.hideWidget = False
        return w