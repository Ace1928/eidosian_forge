from itertools import chain
import re
import six
from genshi.core import escape, Attrs, Markup, QName, StreamEventKind
from genshi.core import START, END, TEXT, XML_DECL, DOCTYPE, START_NS, END_NS, \
def _pop_ns(prefix):
    uris = prefixes.get(prefix)
    uri = uris.pop()
    if not uris:
        del prefixes[prefix]
    if uri not in uris or uri != uris[-1]:
        uri_prefixes = namespaces[uri]
        uri_prefixes.pop()
        if not uri_prefixes:
            del namespaces[uri]
    cache.clear()
    return uri