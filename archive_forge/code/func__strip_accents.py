from __future__ import absolute_import, unicode_literals
import re
import sys
import unicodedata
import six
from pybtex.style.labels import BaseLabelStyle
from pybtex.textutils import abbreviate
def _strip_accents(s):
    return u''.join((c for c in unicodedata.normalize('NFD', s) if not unicodedata.combining(c)))