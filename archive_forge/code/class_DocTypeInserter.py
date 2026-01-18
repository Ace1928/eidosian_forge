from itertools import chain
import re
import six
from genshi.core import escape, Attrs, Markup, QName, StreamEventKind
from genshi.core import START, END, TEXT, XML_DECL, DOCTYPE, START_NS, END_NS, \
class DocTypeInserter(object):
    """A filter that inserts the DOCTYPE declaration in the correct location,
    after the XML declaration.
    """

    def __init__(self, doctype):
        """Initialize the filter.

        :param doctype: DOCTYPE as a string or DocType object.
        """
        if isinstance(doctype, six.string_types):
            doctype = DocType.get(doctype)
        self.doctype_event = (DOCTYPE, doctype, (None, -1, -1))

    def __call__(self, stream):
        doctype_inserted = False
        for kind, data, pos in stream:
            if not doctype_inserted:
                doctype_inserted = True
                if kind is XML_DECL:
                    yield (kind, data, pos)
                    yield self.doctype_event
                    continue
                yield self.doctype_event
            yield (kind, data, pos)
        if not doctype_inserted:
            yield self.doctype_event