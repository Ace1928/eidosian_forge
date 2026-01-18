import copy
import re
from urllib.parse import urlsplit, unquote_plus
from lxml import etree
from lxml.html import defs
from lxml.html import fromstring, XHTML_NAMESPACE
from lxml.html import xhtml_to_html, _transform_result
def _insert_break(word, width, break_character):
    orig_word = word
    result = ''
    while len(word) > width:
        start = word[:width]
        breaks = list(_break_prefer_re.finditer(start))
        if breaks:
            last_break = breaks[-1]
            if last_break.end() > width - 10:
                start = word[:last_break.end()]
        result += start + break_character
        word = word[len(start):]
    result += word
    return result