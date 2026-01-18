import sys
import re
import os
import locale
from functools import reduce
def _XmlEscape(value, attr=False):
    """ Escape a string for inclusion in XML."""

    def replace(match):
        m = match.string[match.start():match.end()]
        if attr and m == "'":
            return m
        return _xml_escape_map[m]
    return _xml_escape_re.sub(replace, value)