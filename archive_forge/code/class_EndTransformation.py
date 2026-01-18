import re
import sys
import six
from genshi.builder import Element
from genshi.core import Stream, Attrs, QName, TEXT, START, END, _ensure, Markup
from genshi.path import Path
class EndTransformation(object):
    """End the current selection."""

    def __call__(self, stream):
        """Apply the transform filter to the marked stream.

        :param stream: the marked event stream to filter
        """
        for mark, event in stream:
            yield (OUTSIDE, event)