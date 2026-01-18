import re
import sys
import six
from genshi.builder import Element
from genshi.core import Stream, Attrs, QName, TEXT, START, END, _ensure, Markup
from genshi.path import Path
class SelectTransformation(object):
    """Select and mark events that match an XPath expression."""

    def __init__(self, path):
        """Create selection.

        :param path: an XPath expression (as string) or a `Path` object
        """
        if not isinstance(path, Path):
            path = Path(path)
        self.path = path

    def __call__(self, stream):
        """Apply the transform filter to the marked stream.

        :param stream: the marked event stream to filter
        """
        namespaces = {}
        variables = {}
        test = self.path.test()
        stream = iter(stream)
        _next = lambda: next(stream)
        for mark, event in stream:
            if mark is None:
                yield (mark, event)
                continue
            result = test(event, namespaces, variables)
            if result is True:
                if event[0] is START:
                    yield (ENTER, event)
                    depth = 1
                    while depth > 0:
                        mark, subevent = _next()
                        if subevent[0] is START:
                            depth += 1
                        elif subevent[0] is END:
                            depth -= 1
                        if depth == 0:
                            yield (EXIT, subevent)
                        else:
                            yield (INSIDE, subevent)
                        test(subevent, namespaces, variables, updateonly=True)
                else:
                    yield (OUTSIDE, event)
            elif isinstance(result, Attrs):
                yield (ATTR, (ATTR, (QName(event[1][0] + '@*'), result), event[2]))
                yield (None, event)
            elif isinstance(result, tuple):
                yield (OUTSIDE, result)
            elif result:
                yield (None, (TEXT, six.text_type(result), (None, -1, -1)))
            else:
                yield (None, event)