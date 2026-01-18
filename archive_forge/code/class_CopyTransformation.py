import re
import sys
import six
from genshi.builder import Element
from genshi.core import Stream, Attrs, QName, TEXT, START, END, _ensure, Markup
from genshi.path import Path
class CopyTransformation(object):
    """Copy selected events into a buffer for later insertion."""

    def __init__(self, buffer, accumulate=False):
        """Create the copy transformation.

        :param buffer: the `StreamBuffer` in which the selection should be
                       stored
        """
        if not accumulate:
            buffer.reset()
        self.buffer = buffer
        self.accumulate = accumulate

    def __call__(self, stream):
        """Apply the transformation to the marked stream.

        :param stream: the marked event stream to filter
        """
        stream = PushBackStream(stream)
        for mark, event in stream:
            if mark:
                if not self.accumulate:
                    self.buffer.reset()
                events = [(mark, event)]
                self.buffer.append(event)
                start = mark
                for mark, event in stream:
                    if start is not ENTER and mark != start:
                        stream.push((mark, event))
                        break
                    events.append((mark, event))
                    self.buffer.append(event)
                    if start is ENTER and mark is EXIT:
                        break
                for i in events:
                    yield i
            else:
                yield (mark, event)