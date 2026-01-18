from itertools import chain
import codecs
from xml.parsers import expat
import six
from six.moves import html_entities as entities, html_parser as html
from genshi.core import Attrs, QName, Stream, stripentities
from genshi.core import START, END, XML_DECL, DOCTYPE, TEXT, START_NS, \
from genshi.compat import StringIO, BytesIO
def _coalesce(stream):
    """Coalesces adjacent TEXT events into a single event."""
    textbuf = []
    textpos = None
    for kind, data, pos in chain(stream, [(None, None, None)]):
        if kind is TEXT:
            textbuf.append(data)
            if textpos is None:
                textpos = pos
        else:
            if textbuf:
                yield (TEXT, ''.join(textbuf), textpos)
                del textbuf[:]
                textpos = None
            if kind:
                yield (kind, data, pos)