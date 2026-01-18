from itertools import chain
import re
import six
from genshi.core import escape, Attrs, Markup, QName, StreamEventKind
from genshi.core import START, END, TEXT, XML_DECL, DOCTYPE, START_NS, END_NS, \
class NamespaceFlattener(object):
    """Output stream filter that removes namespace information from the stream,
    instead adding namespace attributes and prefixes as needed.
    
    :param prefixes: optional mapping of namespace URIs to prefixes
    
    >>> from genshi.input import XML
    >>> xml = XML('''<doc xmlns="NS1" xmlns:two="NS2">
    ...   <two:item/>
    ... </doc>''')
    >>> for kind, data, pos in NamespaceFlattener()(xml):
    ...     print('%s %r' % (kind, data))
    START ('doc', Attrs([('xmlns', 'NS1'), ('xmlns:two', 'NS2')]))
    TEXT '\\n  '
    START ('two:item', Attrs())
    END 'two:item'
    TEXT '\\n'
    END 'doc'
    """

    def __init__(self, prefixes=None, cache=True):
        self.prefixes = {XML_NAMESPACE.uri: 'xml'}
        if prefixes is not None:
            self.prefixes.update(prefixes)
        self.cache = cache

    def __call__(self, stream):
        prefixes = dict([(v, [k]) for k, v in self.prefixes.items()])
        namespaces = {XML_NAMESPACE.uri: ['xml']}
        _emit, _get, cache = _prepare_cache(self.cache)

        def _push_ns(prefix, uri):
            namespaces.setdefault(uri, []).append(prefix)
            prefixes.setdefault(prefix, []).append(uri)
            cache.clear()

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
        ns_attrs = []
        _push_ns_attr = ns_attrs.append

        def _make_ns_attr(prefix, uri):
            return ('xmlns%s' % (prefix and ':%s' % prefix or ''), uri)

        def _gen_prefix():
            val = 0
            while 1:
                val += 1
                yield ('ns%d' % val)
        _prefix_generator = _gen_prefix()
        _gen_prefix = lambda: next(_prefix_generator)
        for kind, data, pos in stream:
            if kind is TEXT and isinstance(data, Markup):
                yield (kind, data, pos)
                continue
            output = _get((kind, data))
            if output is not None:
                yield (kind, output, pos)
            elif kind is START or kind is EMPTY:
                tag, attrs = data
                tagname = tag.localname
                tagns = tag.namespace
                if tagns:
                    if tagns in namespaces:
                        prefix = namespaces[tagns][-1]
                        if prefix:
                            tagname = '%s:%s' % (prefix, tagname)
                    else:
                        _push_ns_attr(('xmlns', tagns))
                        _push_ns('', tagns)
                new_attrs = []
                for attr, value in attrs:
                    attrname = attr.localname
                    attrns = attr.namespace
                    if attrns:
                        if attrns not in namespaces:
                            prefix = _gen_prefix()
                            _push_ns(prefix, attrns)
                            _push_ns_attr(('xmlns:%s' % prefix, attrns))
                        else:
                            prefix = namespaces[attrns][-1]
                        if prefix:
                            attrname = '%s:%s' % (prefix, attrname)
                    new_attrs.append((attrname, value))
                data = _emit(kind, data, (tagname, Attrs(ns_attrs + new_attrs)))
                yield (kind, data, pos)
                del ns_attrs[:]
            elif kind is END:
                tagname = data.localname
                tagns = data.namespace
                if tagns:
                    prefix = namespaces[tagns][-1]
                    if prefix:
                        tagname = '%s:%s' % (prefix, tagname)
                yield (kind, _emit(kind, data, tagname), pos)
            elif kind is START_NS:
                prefix, uri = data
                if uri not in namespaces:
                    prefix = prefixes.get(uri, [prefix])[-1]
                    _push_ns_attr(_make_ns_attr(prefix, uri))
                _push_ns(prefix, uri)
            elif kind is END_NS:
                if data in prefixes:
                    uri = _pop_ns(data)
                    if ns_attrs:
                        attr = _make_ns_attr(data, uri)
                        if attr in ns_attrs:
                            ns_attrs.remove(attr)
            else:
                yield (kind, data, pos)