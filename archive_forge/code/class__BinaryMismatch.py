import operator
from pprint import pformat
import re
import warnings
from ..compat import (
from ..helpers import list_subtract
from ._higherorder import (
from ._impl import (
class _BinaryMismatch(Mismatch):
    """Two things did not match."""

    def __init__(self, actual, mismatch_string, reference, reference_on_right=True):
        self._actual = actual
        self._mismatch_string = mismatch_string
        self._reference = reference
        self._reference_on_right = reference_on_right

    @property
    def expected(self):
        warnings.warn(f'{self.__class__.__name__}.expected deprecated after 1.8.1', DeprecationWarning, stacklevel=2)
        return self._reference

    @property
    def other(self):
        warnings.warn(f'{self.__class__.__name__}.other deprecated after 1.8.1', DeprecationWarning, stacklevel=2)
        return self._actual

    def describe(self):
        actual = repr(self._actual)
        reference = repr(self._reference)
        if len(actual) + len(reference) > 70:
            return '{}:\nreference = {}\nactual    = {}\n'.format(self._mismatch_string, _format(self._reference), _format(self._actual))
        else:
            if self._reference_on_right:
                left, right = (actual, reference)
            else:
                left, right = (reference, actual)
            return f'{left} {self._mismatch_string} {right}'