from itertools import chain
from genshi.core import Attrs, Markup, Namespace, Stream
from genshi.core import START, END, START_NS, END_NS, TEXT, PI, COMMENT
from genshi.input import XMLParser
from genshi.template.base import BadDirectiveError, Template, \
from genshi.template.eval import Suite
from genshi.template.interpolation import interpolate
from genshi.template.directives import *
from genshi.template.text import NewTextTemplate
def _extract_directives(self, stream, namespace, factory):
    depth = 0
    dirmap = {}
    new_stream = []
    ns_prefix = {}
    for kind, data, pos in stream:
        if kind is START:
            tag, attrs = data
            directives = []
            strip = False
            if tag.namespace == namespace:
                cls = factory.get_directive(tag.localname)
                if cls is None:
                    raise BadDirectiveError(tag.localname, self.filepath, pos[1])
                args = dict([(name.localname, value) for name, value in attrs if not name.namespace])
                directives.append((factory.get_directive_index(cls), cls, args, ns_prefix.copy(), pos))
                strip = True
            new_attrs = []
            for name, value in attrs:
                if name.namespace == namespace:
                    cls = factory.get_directive(name.localname)
                    if cls is None:
                        raise BadDirectiveError(name.localname, self.filepath, pos[1])
                    if type(value) is list and len(value) == 1:
                        value = value[0][1]
                    directives.append((factory.get_directive_index(cls), cls, value, ns_prefix.copy(), pos))
                else:
                    new_attrs.append((name, value))
            new_attrs = Attrs(new_attrs)
            if directives:
                directives.sort(key=lambda x: x[0])
                dirmap[depth, tag] = (directives, len(new_stream), strip)
            new_stream.append((kind, (tag, new_attrs), pos))
            depth += 1
        elif kind is END:
            depth -= 1
            new_stream.append((kind, data, pos))
            if (depth, data) in dirmap:
                directives, offset, strip = dirmap.pop((depth, data))
                substream = new_stream[offset:]
                if strip:
                    substream = substream[1:-1]
                new_stream[offset:] = [(SUB, (directives, substream), pos)]
        elif kind is SUB:
            directives, substream = data
            substream = self._extract_directives(substream, namespace, factory)
            if len(substream) == 1 and substream[0][0] is SUB:
                added_directives, substream = substream[0][1]
                directives += added_directives
            new_stream.append((kind, (directives, substream), pos))
        elif kind is START_NS:
            prefix, uri = data
            ns_prefix[prefix] = uri
            if uri != namespace:
                new_stream.append((kind, data, pos))
        elif kind is END_NS:
            uri = ns_prefix.pop(data, None)
            if uri and uri != namespace:
                new_stream.append((kind, data, pos))
        else:
            new_stream.append((kind, data, pos))
    return new_stream