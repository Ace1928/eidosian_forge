import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
class MinLength(FancyValidator):
    """
    Invalid if the value is shorter than `minlength`.  Uses len(), so
    it can work for strings, lists, or anything with length.  Note
    that you **must** use ``not_empty=True`` if you don't want to
    accept empty values -- empty values are not tested for length.

    Examples::

        >>> min5 = MinLength(5)
        >>> min5.to_python('12345')
        '12345'
        >>> min5.from_python('12345')
        '12345'
        >>> min5.to_python('1234')
        Traceback (most recent call last):
          ...
        Invalid: Enter a value at least 5 characters long
        >>> min5(accept_python=False).from_python('1234')
        Traceback (most recent call last):
          ...
        Invalid: Enter a value at least 5 characters long
        >>> min5.to_python([1, 2, 3, 4, 5])
        [1, 2, 3, 4, 5]
        >>> min5.to_python([1, 2, 3])
        Traceback (most recent call last):
          ...
        Invalid: Enter a value at least 5 characters long
        >>> min5.to_python(5)
        Traceback (most recent call last):
          ...
        Invalid: Invalid value (value with length expected)

    """
    __unpackargs__ = ('minLength',)
    messages = dict(tooShort=_('Enter a value at least %(minLength)i characters long'), invalid=_('Invalid value (value with length expected)'))

    def _validate_python(self, value, state):
        try:
            if len(value) < self.minLength:
                raise Invalid(self.message('tooShort', state, minLength=self.minLength), value, state)
        except TypeError:
            raise Invalid(self.message('invalid', state), value, state)