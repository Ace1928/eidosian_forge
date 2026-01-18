import re
import sys
import six
from genshi.builder import Element
from genshi.core import Stream, Attrs, QName, TEXT, START, END, _ensure, Markup
from genshi.path import Path
class AfterTransformation(InjectorTransformation):
    """Insert content after selection."""

    def __call__(self, stream):
        """Apply the transform filter to the marked stream.

        :param stream: The marked event stream to filter
        """
        stream = PushBackStream(stream)
        for mark, event in stream:
            yield (mark, event)
            if mark:
                start = mark
                for mark, event in stream:
                    if start is not ENTER and mark != start:
                        stream.push((mark, event))
                        break
                    yield (mark, event)
                    if start is ENTER and mark is EXIT:
                        break
                for subevent in self._inject():
                    yield subevent