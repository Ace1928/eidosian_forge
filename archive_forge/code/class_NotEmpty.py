import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
class NotEmpty(FancyValidator):
    """
    Invalid if value is empty (empty string, empty list, etc).

    Generally for objects that Python considers false, except zero
    which is not considered invalid.

    Examples::

        >>> ne = NotEmpty(messages=dict(empty='enter something'))
        >>> ne.to_python('')
        Traceback (most recent call last):
          ...
        Invalid: enter something
        >>> ne.to_python(0)
        0
    """
    not_empty = True
    messages = dict(empty=_('Please enter a value'))

    def _validate_python(self, value, state):
        if value == 0:
            return value
        if not value:
            raise Invalid(self.message('empty', state), value, state)