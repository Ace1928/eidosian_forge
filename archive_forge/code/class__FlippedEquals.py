import operator
from pprint import pformat
import re
import warnings
from ..compat import (
from ..helpers import list_subtract
from ._higherorder import (
from ._impl import (
class _FlippedEquals:
    """Matches if the items are equal.

    Exactly like ``Equals`` except that the short mismatch message is "
    $reference != $actual" rather than "$actual != $reference". This allows
    for ``TestCase.assertEqual`` to use a matcher but still have the order of
    items in the error message align with the order of items in the call to
    the assertion.
    """

    def __init__(self, expected):
        self._expected = expected

    def match(self, other):
        mismatch = Equals(self._expected).match(other)
        if not mismatch:
            return None
        return _BinaryMismatch(other, '!=', self._expected, False)