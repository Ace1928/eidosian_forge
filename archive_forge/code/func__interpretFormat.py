from ...Qt import QtCore, QtWidgets
from ..Parameter import Parameter
from .basetypes import WidgetParameterItem
def _interpretFormat(self, fmt=None):
    fmt = fmt or self.opts.get('format')
    if hasattr(QtCore.Qt.DateFormat, fmt):
        fmt = getattr(QtCore.Qt.DateFormat, fmt)
    return fmt