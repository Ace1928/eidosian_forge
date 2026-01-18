import re
import sys
import six
from genshi.builder import Element
from genshi.core import Stream, Attrs, QName, TEXT, START, END, _ensure, Markup
from genshi.path import Path
class RenameTransformation(object):
    """Rename matching elements."""

    def __init__(self, name):
        """Create the transform.

        :param name: New element name.
        """
        self.name = QName(name)

    def __call__(self, stream):
        """Apply the transform filter to the marked stream.

        :param stream: The marked event stream to filter
        """
        for mark, (kind, data, pos) in stream:
            if mark is ENTER:
                data = (self.name, data[1])
            elif mark is EXIT:
                data = self.name
            yield (mark, (kind, data, pos))