import codecs
from html.entities import codepoint2name
from html.entities import name2codepoint
import re
from urllib.parse import quote_plus
import markupsafe
def __escape(self, m):
    codepoint = ord(m.group())
    try:
        return self.codepoint2entity[codepoint]
    except (KeyError, IndexError):
        return '&#x%X;' % codepoint