import re
import sys
import six
from genshi.builder import Element
from genshi.core import Stream, Attrs, QName, TEXT, START, END, _ensure, Markup
from genshi.path import Path
class PrependTransformation(InjectorTransformation):
    """Prepend content to the inside of selected elements."""

    def __call__(self, stream):
        """Apply the transform filter to the marked stream.

        :param stream: The marked event stream to filter
        """
        for mark, event in stream:
            yield (mark, event)
            if mark is ENTER:
                for subevent in self._inject():
                    yield subevent