import re
import numpy as np
class IntFormat:

    @classmethod
    def from_number(cls, n, min=None):
        """Given an integer, returns a "reasonable" IntFormat instance to represent
        any number between 0 and n if n > 0, -n and n if n < 0

        Parameters
        ----------
        n : int
            max number one wants to be able to represent
        min : int
            minimum number of characters to use for the format

        Returns
        -------
        res : IntFormat
            IntFormat instance with reasonable (see Notes) computed width

        Notes
        -----
        Reasonable should be understood as the minimal string length necessary
        without losing precision. For example, IntFormat.from_number(1) will
        return an IntFormat instance of width 2, so that any 0 and 1 may be
        represented as 1-character strings without loss of information.
        """
        width = number_digits(n) + 1
        if n < 0:
            width += 1
        repeat = 80 // width
        return cls(width, min, repeat=repeat)

    def __init__(self, width, min=None, repeat=None):
        self.width = width
        self.repeat = repeat
        self.min = min

    def __repr__(self):
        r = 'IntFormat('
        if self.repeat:
            r += '%d' % self.repeat
        r += 'I%d' % self.width
        if self.min:
            r += '.%d' % self.min
        return r + ')'

    @property
    def fortran_format(self):
        r = '('
        if self.repeat:
            r += '%d' % self.repeat
        r += 'I%d' % self.width
        if self.min:
            r += '.%d' % self.min
        return r + ')'

    @property
    def python_format(self):
        return '%' + str(self.width) + 'd'