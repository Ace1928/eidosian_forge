import sys
import encodings
import encodings.aliases
import re
import _collections_abc
from builtins import str as _builtin_str
import functools
def _build_localename(localetuple):
    """ Builds a locale code from the given tuple (language code,
        encoding).

        No aliasing or normalizing takes place.

    """
    try:
        language, encoding = localetuple
        if language is None:
            language = 'C'
        if encoding is None:
            return language
        else:
            return language + '.' + encoding
    except (TypeError, ValueError):
        raise TypeError('Locale must be None, a string, or an iterable of two strings -- language code, encoding.') from None