import re
import sys
import six
from genshi.builder import Element
from genshi.core import Stream, Attrs, QName, TEXT, START, END, _ensure, Markup
from genshi.path import Path
def _mark(self, stream):
    for event in stream:
        yield (OUTSIDE, event)