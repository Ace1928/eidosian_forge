import gzip
import re
import secrets
import unicodedata
from gzip import GzipFile
from gzip import compress as gzip_compress
from io import BytesIO
from django.core.exceptions import SuspiciousFileOperation
from django.utils.functional import SimpleLazyObject, keep_lazy_text, lazy
from django.utils.regex_helper import _lazy_re_compile
from django.utils.translation import gettext as _
from django.utils.translation import gettext_lazy, pgettext
def _truncate_html(self, length, truncate, text, truncate_len, words):
    """
        Truncate HTML to a certain number of chars (not counting tags and
        comments), or, if words is True, then to a certain number of words.
        Close opened tags if they were correctly closed in the given HTML.

        Preserve newlines in the HTML.
        """
    if words and length <= 0:
        return ''
    size_limited = False
    if len(text) > self.MAX_LENGTH_HTML:
        text = text[:self.MAX_LENGTH_HTML]
        size_limited = True
    html4_singlets = ('br', 'col', 'link', 'base', 'img', 'param', 'area', 'hr', 'input')
    pos = 0
    end_text_pos = 0
    current_len = 0
    open_tags = []
    regex = re_words if words else re_chars
    while current_len <= length:
        m = regex.search(text, pos)
        if not m:
            break
        pos = m.end(0)
        if m[1]:
            current_len += 1
            if current_len == truncate_len:
                end_text_pos = pos
            continue
        tag = re_tag.match(m[0])
        if not tag or current_len >= truncate_len:
            continue
        closing_tag, tagname, self_closing = tag.groups()
        tagname = tagname.lower()
        if self_closing or tagname in html4_singlets:
            pass
        elif closing_tag:
            try:
                i = open_tags.index(tagname)
            except ValueError:
                pass
            else:
                open_tags = open_tags[i + 1:]
        else:
            open_tags.insert(0, tagname)
    truncate_text = add_truncation_text('', truncate)
    if current_len <= length:
        if size_limited and truncate_text:
            text += truncate_text
        return text
    out = text[:end_text_pos]
    if truncate_text:
        out += truncate_text
    for tag in open_tags:
        out += '</%s>' % tag
    return out