import re
from html import escape
from html.entities import name2codepoint
from html.parser import HTMLParser
def _sub_entityref(self, match):
    name = match.group(1)
    if name not in name2codepoint:
        return match.group(0)
    return chr(name2codepoint[name])