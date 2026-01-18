import builtins
from ... import functions as fn
from ... import icons
from ...Qt import QtCore, QtWidgets
from ..Parameter import Parameter
from ..ParameterItem import ParameterItem
class SimpleParameter(Parameter):
    """
    Parameter representing a single value.

    This parameter is backed by :class:`~pyqtgraph.parametertree.parameterTypes.basetypes.WidgetParameterItem`
     to represent the following parameter names through various subclasses:

      - 'int'
      - 'float'
      - 'bool'
      - 'str'
      - 'color'
      - 'colormap'
    """

    @property
    def itemClass(self):
        from .bool import BoolParameterItem
        from .numeric import NumericParameterItem
        from .str import StrParameterItem
        return {'bool': BoolParameterItem, 'int': NumericParameterItem, 'float': NumericParameterItem, 'str': StrParameterItem}[self.opts['type']]

    def _interpretValue(self, v):
        typ = self.opts['type']

        def _missing_interp(v):
            return v
        interpreter = getattr(builtins, typ, _missing_interp)
        return interpreter(v)