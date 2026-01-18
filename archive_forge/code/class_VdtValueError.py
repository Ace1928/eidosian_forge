import re
import sys
from pprint import pprint
class VdtValueError(ValidateError):
    """The value supplied was of the correct type, but was not an allowed value."""

    def __init__(self, value):
        """
        >>> raise VdtValueError('jedi')
        Traceback (most recent call last):
        VdtValueError: the value "jedi" is unacceptable.
        """
        ValidateError.__init__(self, 'the value "%s" is unacceptable.' % (value,))