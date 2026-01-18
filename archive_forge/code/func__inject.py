import re
import sys
import six
from genshi.builder import Element
from genshi.core import Stream, Attrs, QName, TEXT, START, END, _ensure, Markup
from genshi.path import Path
def _inject(self):
    content = self.content
    if hasattr(content, '__call__'):
        content = content()
    for event in _ensure(content):
        yield (None, event)