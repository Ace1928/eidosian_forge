from heapq import nlargest as _nlargest
from collections import namedtuple as _namedtuple
from types import GenericAlias
import re
def _format_line(self, side, flag, linenum, text):
    """Returns HTML markup of "from" / "to" text lines

        side -- 0 or 1 indicating "from" or "to" text
        flag -- indicates if difference on line
        linenum -- line number (used for line number column)
        text -- line text to be marked up
        """
    try:
        linenum = '%d' % linenum
        id = ' id="%s%s"' % (self._prefix[side], linenum)
    except TypeError:
        id = ''
    text = text.replace('&', '&amp;').replace('>', '&gt;').replace('<', '&lt;')
    text = text.replace(' ', '&nbsp;').rstrip()
    return '<td class="diff_header"%s>%s</td><td nowrap="nowrap">%s</td>' % (id, linenum, text)