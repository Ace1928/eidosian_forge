import re
import numpy as np
class ExpFormat:

    @classmethod
    def from_number(cls, n, min=None):
        """Given a float number, returns a "reasonable" ExpFormat instance to
        represent any number between -n and n.

        Parameters
        ----------
        n : float
            max number one wants to be able to represent
        min : int
            minimum number of characters to use for the format

        Returns
        -------
        res : ExpFormat
            ExpFormat instance with reasonable (see Notes) computed width

        Notes
        -----
        Reasonable should be understood as the minimal string length necessary
        to avoid losing precision.
        """
        finfo = np.finfo(n.dtype)
        n_prec = finfo.precision + 1
        n_exp = number_digits(np.max(np.abs([finfo.maxexp, finfo.minexp])))
        width = 1 + 1 + n_prec + 1 + n_exp + 1
        if n < 0:
            width += 1
        repeat = int(np.floor(80 / width))
        return cls(width, n_prec, min, repeat=repeat)

    def __init__(self, width, significand, min=None, repeat=None):
        """        Parameters
        ----------
        width : int
            number of characters taken by the string (includes space).
        """
        self.width = width
        self.significand = significand
        self.repeat = repeat
        self.min = min

    def __repr__(self):
        r = 'ExpFormat('
        if self.repeat:
            r += '%d' % self.repeat
        r += 'E%d.%d' % (self.width, self.significand)
        if self.min:
            r += 'E%d' % self.min
        return r + ')'

    @property
    def fortran_format(self):
        r = '('
        if self.repeat:
            r += '%d' % self.repeat
        r += 'E%d.%d' % (self.width, self.significand)
        if self.min:
            r += 'E%d' % self.min
        return r + ')'

    @property
    def python_format(self):
        return '%' + str(self.width - 1) + '.' + str(self.significand) + 'E'