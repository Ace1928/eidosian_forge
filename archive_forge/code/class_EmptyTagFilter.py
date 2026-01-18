from itertools import chain
import re
import six
from genshi.core import escape, Attrs, Markup, QName, StreamEventKind
from genshi.core import START, END, TEXT, XML_DECL, DOCTYPE, START_NS, END_NS, \
class EmptyTagFilter(object):
    """Combines `START` and `STOP` events into `EMPTY` events for elements that
    have no contents.
    """
    EMPTY = StreamEventKind('EMPTY')

    def __call__(self, stream):
        prev = (None, None, None)
        for ev in stream:
            if prev[0] is START:
                if ev[0] is END:
                    prev = (EMPTY, prev[1], prev[2])
                    yield prev
                    continue
                else:
                    yield prev
            if ev[0] is not START:
                yield ev
            prev = ev