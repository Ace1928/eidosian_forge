from lxml import etree
import sys
import re
import doctest
def _looks_like_markup(self, s):
    s = s.strip()
    return s.startswith('<') and (not _repr_re.search(s))