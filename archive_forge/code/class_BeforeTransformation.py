import re
import sys
import six
from genshi.builder import Element
from genshi.core import Stream, Attrs, QName, TEXT, START, END, _ensure, Markup
from genshi.path import Path
class BeforeTransformation(InjectorTransformation):
    """Insert content before selection."""

    def __call__(self, stream):
        """Apply the transform filter to the marked stream.

        :param stream: The marked event stream to filter
        """
        stream = PushBackStream(stream)
        for mark, event in stream:
            if mark is not None:
                start = mark
                for subevent in self._inject():
                    yield subevent
                yield (mark, event)
                for mark, event in stream:
                    if mark != start and start is not ENTER:
                        stream.push((mark, event))
                        break
                    yield (mark, event)
                    if start is ENTER and mark is EXIT:
                        break
            else:
                yield (mark, event)