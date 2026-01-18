import re
import sys
from pprint import pprint
class VdtValueTooBigError(VdtValueError):
    """The value supplied was of the correct type, but was too big."""

    def __init__(self, value):
        """
        >>> raise VdtValueTooBigError('1')
        Traceback (most recent call last):
        VdtValueTooBigError: the value "1" is too big.
        """
        ValidateError.__init__(self, 'the value "%s" is too big.' % (value,))