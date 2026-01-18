from itertools import chain
import re
import six
from genshi.core import escape, Attrs, Markup, QName, StreamEventKind
from genshi.core import START, END, TEXT, XML_DECL, DOCTYPE, START_NS, END_NS, \
class HTMLSerializer(XHTMLSerializer):
    """Produces HTML text from an event stream.
    
    >>> from genshi.builder import tag
    >>> elem = tag.div(tag.a(href='foo'), tag.br, tag.hr(noshade=True))
    >>> print(''.join(HTMLSerializer()(elem.generate())))
    <div><a href="foo"></a><br><hr noshade></div>
    """
    _NOESCAPE_ELEMS = frozenset([QName('script'), QName('http://www.w3.org/1999/xhtml}script'), QName('style'), QName('http://www.w3.org/1999/xhtml}style')])

    def __init__(self, doctype=None, strip_whitespace=True, cache=True):
        """Initialize the HTML serializer.
        
        :param doctype: a ``(name, pubid, sysid)`` tuple that represents the
                        DOCTYPE declaration that should be included at the top
                        of the generated output
        :param strip_whitespace: whether extraneous whitespace should be
                                 stripped from the output
        :param cache: whether to cache the text output per event, which
                      improves performance for repetitive markup
        :note: Changed in 0.6: The `cache` parameter was added
        """
        super(HTMLSerializer, self).__init__(doctype, False)
        self.filters = [EmptyTagFilter()]
        if strip_whitespace:
            self.filters.append(WhitespaceFilter(self._PRESERVE_SPACE, self._NOESCAPE_ELEMS))
        self.filters.append(NamespaceFlattener(prefixes={'http://www.w3.org/1999/xhtml': ''}, cache=cache))
        if doctype:
            self.filters.append(DocTypeInserter(doctype))
        self.cache = True

    def __call__(self, stream):
        boolean_attrs = self._BOOLEAN_ATTRS
        empty_elems = self._EMPTY_ELEMS
        noescape_elems = self._NOESCAPE_ELEMS
        have_doctype = False
        noescape = False
        _emit, _get = self._prepare_cache()
        for filter_ in self.filters:
            stream = filter_(stream)
        for kind, data, _ in stream:
            if kind is TEXT and isinstance(data, Markup):
                yield data
                continue
            output = _get((kind, data))
            if output is not None:
                yield output
                if (kind is START or kind is EMPTY) and data[0] in noescape_elems:
                    noescape = True
                elif kind is END:
                    noescape = False
            elif kind is START or kind is EMPTY:
                tag, attrib = data
                buf = ['<', tag]
                for attr, value in attrib:
                    if attr in boolean_attrs:
                        if value:
                            buf += [' ', attr]
                    elif ':' in attr:
                        if attr == 'xml:lang' and 'lang' not in attrib:
                            buf += [' lang="', escape(value), '"']
                    elif attr != 'xmlns':
                        buf += [' ', attr, '="', escape(value), '"']
                buf.append('>')
                if kind is EMPTY:
                    if tag not in empty_elems:
                        buf.append('</%s>' % tag)
                yield _emit(kind, data, Markup(''.join(buf)))
                if tag in noescape_elems:
                    noescape = True
            elif kind is END:
                yield _emit(kind, data, Markup('</%s>' % data))
                noescape = False
            elif kind is TEXT:
                if noescape:
                    yield _emit(kind, data, data)
                else:
                    yield _emit(kind, data, escape(data, quotes=False))
            elif kind is COMMENT:
                yield _emit(kind, data, Markup('<!--%s-->' % data))
            elif kind is DOCTYPE and (not have_doctype):
                name, pubid, sysid = data
                buf = ['<!DOCTYPE %s']
                if pubid:
                    buf.append(' PUBLIC "%s"')
                elif sysid:
                    buf.append(' SYSTEM')
                if sysid:
                    buf.append(' "%s"')
                buf.append('>\n')
                yield (Markup(''.join(buf)) % tuple([p for p in data if p]))
                have_doctype = True
            elif kind is PI:
                yield _emit(kind, data, Markup('<?%s %s?>' % data))