import warnings
from collections import OrderedDict
from ... import functions as fn
from ...Qt import QtWidgets
from ..Parameter import Parameter
from .basetypes import WidgetParameterItem
class ListParameter(Parameter):
    """Parameter with a list of acceptable values.

    By default, this parameter is represtented by a :class:`ListParameterItem`,
    displaying a combo box to select a value from the list.

    In addition to the generic :class:`~pyqtgraph.parametertree.Parameter`
    options, this parameter type accepts a ``limits`` argument specifying the
    list of allowed values.

    The values may generally be of any data type, as long as they can be
    represented as a string. If the string representation provided is
    undesirable, the values may be given as a dictionary mapping the desired
    string representation to the value.
    """
    itemClass = ListParameterItem

    def __init__(self, **opts):
        self.forward = OrderedDict()
        self.reverse = ([], [])
        if opts.get('limits', None) is None:
            opts['limits'] = []
        Parameter.__init__(self, **opts)
        self.setLimits(opts['limits'])

    def setLimits(self, limits):
        """Change the list of allowed values."""
        self.forward, self.reverse = self.mapping(limits)
        Parameter.setLimits(self, limits)
        if self.hasValue():
            curVal = self.value()
            if len(self.reverse[0]) > 0 and (not any((fn.eq(curVal, limVal) for limVal in self.reverse[0]))):
                self.setValue(self.reverse[0][0])

    @staticmethod
    def mapping(limits):
        forward = OrderedDict()
        reverse = ([], [])
        if not isinstance(limits, dict):
            limits = {str(l): l for l in limits}
        for k, v in limits.items():
            forward[k] = v
            reverse[0].append(v)
            reverse[1].append(k)
        return (forward, reverse)