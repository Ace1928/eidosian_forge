from ...Qt import QtGui, QtWidgets
from ..Parameter import Parameter
from .basetypes import WidgetParameterItem
class FontParameter(Parameter):
    """
    Creates and controls a QFont value. Be careful when selecting options from the font dropdown. since not all
    fonts are available on all systems
    """
    itemClass = FontParameterItem

    def _interpretValue(self, v):
        if isinstance(v, str):
            newVal = QtGui.QFont()
            if not newVal.fromString(v):
                raise ValueError(f'Error parsing font "{v}"')
            v = newVal
        return v

    def saveState(self, filter=None):
        state = super().saveState(filter)
        state['value'] = state['value'].toString()
        return state