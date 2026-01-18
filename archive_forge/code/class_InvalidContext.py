import math as _math
import numbers as _numbers
import sys
import contextvars
import re
class InvalidContext(InvalidOperation):
    """Invalid context.  Unknown rounding, for example.

    This occurs and signals invalid-operation if an invalid context was
    detected during an operation.  This can occur if contexts are not checked
    on creation and either the precision exceeds the capability of the
    underlying concrete representation or an unknown or unsupported rounding
    was specified.  These aspects of the context need only be checked when
    the values are required to be used.  The result is [0,qNaN].
    """

    def handle(self, context, *args):
        return _NaN