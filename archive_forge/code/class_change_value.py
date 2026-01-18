from time import time
from ..api import Any, DelegatesTo, HasTraits, Int, Range
class change_value(int_value):

    def _value_changed(self, old, new):
        pass