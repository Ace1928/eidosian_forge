import sys
import encodings
import encodings.aliases
import re
import _collections_abc
from builtins import str as _builtin_str
import functools
def _parse_localename(localename):
    """ Parses the locale code for localename and returns the
        result as tuple (language code, encoding).

        The localename is normalized and passed through the locale
        alias engine. A ValueError is raised in case the locale name
        cannot be parsed.

        The language code corresponds to RFC 1766.  code and encoding
        can be None in case the values cannot be determined or are
        unknown to this implementation.

    """
    code = normalize(localename)
    if '@' in code:
        code, modifier = code.split('@', 1)
        if modifier == 'euro' and '.' not in code:
            return (code, 'iso-8859-15')
    if '.' in code:
        return tuple(code.split('.')[:2])
    elif code == 'C':
        return (None, None)
    elif code == 'UTF-8':
        return (None, 'UTF-8')
    raise ValueError('unknown locale: %s' % localename)