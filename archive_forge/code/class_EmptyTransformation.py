import re
import sys
import six
from genshi.builder import Element
from genshi.core import Stream, Attrs, QName, TEXT, START, END, _ensure, Markup
from genshi.path import Path
class EmptyTransformation(object):
    """Empty selected elements of all content."""

    def __call__(self, stream):
        """Apply the transform filter to the marked stream.

        :param stream: the marked event stream to filter
        """
        for mark, event in stream:
            yield (mark, event)
            if mark is ENTER:
                for mark, event in stream:
                    if mark is EXIT:
                        yield (mark, event)
                        break