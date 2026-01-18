import re
from html import escape
from html.entities import name2codepoint
from html.parser import HTMLParser
def _sub_charref(self, match):
    num = match.group(1)
    if num.lower().startswith('x'):
        num = int(num[1:], 16)
    else:
        num = int(num)
    return chr(num)