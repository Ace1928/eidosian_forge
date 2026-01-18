from itertools import chain
import re
import six
from genshi.core import escape, Attrs, Markup, QName, StreamEventKind
from genshi.core import START, END, TEXT, XML_DECL, DOCTYPE, START_NS, END_NS, \
def _gen_prefix():
    val = 0
    while 1:
        val += 1
        yield ('ns%d' % val)