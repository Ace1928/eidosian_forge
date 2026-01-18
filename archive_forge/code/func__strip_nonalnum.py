from __future__ import absolute_import, unicode_literals
import re
import sys
import unicodedata
import six
from pybtex.style.labels import BaseLabelStyle
from pybtex.textutils import abbreviate
def _strip_nonalnum(parts):
    """Strip all non-alphanumerical characters from a list of strings.

    >>> print(_strip_nonalnum([u"ÅA. B. Testing 12+}[.@~_", u" 3%"]))
    AABTesting123
    """
    s = u''.join(parts)
    return _nonalnum_pattern.sub(u'', _strip_accents(s))