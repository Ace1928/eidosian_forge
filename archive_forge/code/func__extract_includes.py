from itertools import chain
from genshi.core import Attrs, Markup, Namespace, Stream
from genshi.core import START, END, START_NS, END_NS, TEXT, PI, COMMENT
from genshi.input import XMLParser
from genshi.template.base import BadDirectiveError, Template, \
from genshi.template.eval import Suite
from genshi.template.interpolation import interpolate
from genshi.template.directives import *
from genshi.template.text import NewTextTemplate
def _extract_includes(self, stream):
    streams = [[]]
    prefixes = {}
    fallbacks = []
    includes = []
    xinclude_ns = Namespace(self.XINCLUDE_NAMESPACE)
    for kind, data, pos in stream:
        stream = streams[-1]
        if kind is START:
            tag, attrs = data
            if tag in xinclude_ns:
                if tag.localname == 'include':
                    include_href = attrs.get('href')
                    if not include_href:
                        raise TemplateSyntaxError('Include misses required attribute "href"', self.filepath, *pos[1:])
                    includes.append((include_href, attrs.get('parse')))
                    streams.append([])
                elif tag.localname == 'fallback':
                    streams.append([])
                    fallbacks.append(streams[-1])
            else:
                stream.append((kind, (tag, attrs), pos))
        elif kind is END:
            if fallbacks and data == xinclude_ns['fallback']:
                fallback_stream = streams.pop()
                assert fallback_stream is fallbacks[-1]
            elif data == xinclude_ns['include']:
                fallback = None
                if len(fallbacks) == len(includes):
                    fallback = fallbacks.pop()
                streams.pop()
                stream = streams[-1]
                href, parse = includes.pop()
                try:
                    cls = {'xml': MarkupTemplate, 'text': NewTextTemplate}.get(parse) or self.__class__
                except KeyError:
                    raise TemplateSyntaxError('Invalid value for "parse" attribute of include', self.filepath, *pos[1:])
                stream.append((INCLUDE, (href, cls, fallback), pos))
            else:
                stream.append((kind, data, pos))
        elif kind is START_NS and data[1] == xinclude_ns:
            prefixes[data[0]] = data[1]
        elif kind is END_NS and data in prefixes:
            prefixes.pop(data)
        else:
            stream.append((kind, data, pos))
    assert len(streams) == 1
    return streams[0]