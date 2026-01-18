import re
import sys
from pprint import pprint
class VdtValueTooLongError(VdtValueError):
    """The value supplied was of the correct type, but was too long."""

    def __init__(self, value):
        """
        >>> raise VdtValueTooLongError('jedie')
        Traceback (most recent call last):
        VdtValueTooLongError: the value "jedie" is too long.
        """
        ValidateError.__init__(self, 'the value "%s" is too long.' % (value,))