from itertools import chain
import re
import six
from genshi.core import escape, Attrs, Markup, QName, StreamEventKind
from genshi.core import START, END, TEXT, XML_DECL, DOCTYPE, START_NS, END_NS, \
def _make_ns_attr(prefix, uri):
    return ('xmlns%s' % (prefix and ':%s' % prefix or ''), uri)