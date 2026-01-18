from ...Qt import QtWidgets
from ..Parameter import Parameter
from .basetypes import WidgetParameterItem
class ProgressBarParameter(Parameter):
    """
    Displays a progress bar whose value can be set between 0 and 100
    """
    itemClass = ProgressBarParameterItem