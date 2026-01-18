import re
import sys
import six
from genshi.builder import Element
from genshi.core import Stream, Attrs, QName, TEXT, START, END, _ensure, Markup
from genshi.path import Path
def _unmark(self, stream):
    for mark, event in stream:
        kind = event[0]
        if not (kind is None or kind is ATTR or kind is BREAK):
            yield event