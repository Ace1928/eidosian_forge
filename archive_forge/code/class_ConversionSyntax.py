import math as _math
import numbers as _numbers
import sys
import contextvars
import re
class ConversionSyntax(InvalidOperation):
    """Trying to convert badly formed string.

    This occurs and signals invalid-operation if a string is being
    converted to a number and it does not conform to the numeric string
    syntax.  The result is [0,qNaN].
    """

    def handle(self, context, *args):
        return _NaN