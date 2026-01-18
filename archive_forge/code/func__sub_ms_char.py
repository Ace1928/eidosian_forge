from html.entities import codepoint2name
from collections import defaultdict
import codecs
import re
import logging
import string
from html.entities import html5
def _sub_ms_char(self, match):
    """Changes a MS smart quote character to an XML or HTML
        entity, or an ASCII character."""
    orig = match.group(1)
    if self.smart_quotes_to == 'ascii':
        sub = self.MS_CHARS_TO_ASCII.get(orig).encode()
    else:
        sub = self.MS_CHARS.get(orig)
        if type(sub) == tuple:
            if self.smart_quotes_to == 'xml':
                sub = '&#x'.encode() + sub[1].encode() + ';'.encode()
            else:
                sub = '&'.encode() + sub[0].encode() + ';'.encode()
        else:
            sub = sub.encode()
    return sub