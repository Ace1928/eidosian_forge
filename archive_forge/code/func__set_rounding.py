import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _set_rounding(self, type):
    """Sets the rounding type.

        Sets the rounding type, and returns the current (previous)
        rounding type.  Often used like:

        context = context.copy()
        # so you don't change the calling context
        # if an error occurs in the middle.
        rounding = context._set_rounding(ROUND_UP)
        val = self.__sub__(other, context=context)
        context._set_rounding(rounding)

        This will make it round up for that operation.
        """
    rounding = self.rounding
    self.rounding = type
    return rounding