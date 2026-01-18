from array import array
import struct
import sys
import traceback
import types
from Xlib import X
from Xlib.support import lock
class LengthField(Field):
    """A LengthField stores the length of some other Field whose size
    may vary, e.g. List and String8.

    Its name should be the same as the name of the field whose size
    it stores.

    The lf.get_binary_value() method of LengthFields is not used, instead
    a lf.get_binary_length() should be provided.

    Unless LengthField.get_binary_length() is overridden in child classes,
    there should also be a lf.calc_length().
    """
    structcode = 'L'
    structvalues = 1

    def calc_length(self, length):
        """newlen = lf.calc_length(length)

        Return a new length NEWLEN based on the provided LENGTH.
        """
        return length