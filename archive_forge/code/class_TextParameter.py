from ...Qt import QtCore, QtWidgets
from ..Parameter import Parameter
from .basetypes import WidgetParameterItem
class TextParameter(Parameter):
    """Editable string, displayed as large text box in the tree."""
    itemClass = TextParameterItem