from ... import functions as fn
from ...Qt import QtWidgets
from ...SignalProxy import SignalProxy
from ..ParameterItem import ParameterItem
from . import BoolParameterItem, SimpleParameter
from .basetypes import Emitter, GroupParameter, GroupParameterItem, WidgetParameterItem
from .list import ListParameter
def _intersectionWithLimits(self, values: list):
    """
        Returns the (names, values) from limits that intersect with ``values``.
        """
    allowedNames = []
    allowedValues = []
    for val in values:
        for limitName, limitValue in zip(*self.reverse):
            if fn.eq(limitValue, val):
                allowedNames.append(limitName)
                allowedValues.append(val)
                break
    return (allowedNames, allowedValues)