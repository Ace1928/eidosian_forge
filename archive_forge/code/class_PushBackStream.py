import re
import sys
import six
from genshi.builder import Element
from genshi.core import Stream, Attrs, QName, TEXT, START, END, _ensure, Markup
from genshi.path import Path
class PushBackStream(object):
    """Allows a single event to be pushed back onto the stream and re-consumed.
    """

    def __init__(self, stream):
        self.stream = iter(stream)
        self.peek = None

    def push(self, event):
        assert self.peek is None
        self.peek = event

    def __iter__(self):
        while True:
            if self.peek is not None:
                peek = self.peek
                self.peek = None
                yield peek
            else:
                try:
                    event = next(self.stream)
                    yield event
                except StopIteration:
                    if self.peek is None:
                        return