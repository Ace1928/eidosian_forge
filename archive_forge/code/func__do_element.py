import string
from xml.dom import Node
def _do_element(self, node, initial_other_attrs=[], unused=None):
    """_do_element(self, node, initial_other_attrs = [], unused = {}) -> None
        Process an element (and its children)."""
    ns_parent, ns_rendered, xml_attrs = (self.state[0], self.state[1].copy(), self.state[2].copy())
    ns_unused_inherited = unused
    if unused is None:
        ns_unused_inherited = self.state[3].copy()
    ns_local = ns_parent.copy()
    inclusive = _inclusive(self)
    xml_attrs_local = {}
    other_attrs = []
    in_subset = _in_subset(self.subset, node)
    for a in initial_other_attrs + _attrs(node):
        if a.namespaceURI == XMLNS.BASE:
            n = a.nodeName
            if n == 'xmlns:':
                n = 'xmlns'
            ns_local[n] = a.nodeValue
        elif a.namespaceURI == XMLNS.XML:
            if inclusive or (in_subset and _in_subset(self.subset, a)):
                xml_attrs_local[a.nodeName] = a
        elif _in_subset(self.subset, a):
            other_attrs.append(a)
        xml_attrs.update(xml_attrs_local)
    W, name = (self.write, None)
    if in_subset:
        name = node.nodeName
        if not inclusive:
            if node.prefix is not None:
                prefix = 'xmlns:%s' % node.prefix
            else:
                prefix = 'xmlns'
            if not ns_rendered.has_key(prefix) and (not ns_local.has_key(prefix)):
                if not ns_unused_inherited.has_key(prefix):
                    raise RuntimeError('For exclusive c14n, unable to map prefix "%s" in %s' % (prefix, node))
                ns_local[prefix] = ns_unused_inherited[prefix]
                del ns_unused_inherited[prefix]
        W('<')
        W(name)
        ns_to_render = []
        for n, v in ns_local.items():
            if n == 'xmlns' and v in [XMLNS.BASE, ''] and (ns_rendered.get('xmlns') in [XMLNS.BASE, '', None]):
                continue
            if n in ['xmlns:xml', 'xml'] and v in ['http://www.w3.org/XML/1998/namespace']:
                continue
            if (n, v) not in ns_rendered.items():
                if inclusive or _utilized(n, node, other_attrs, self.unsuppressedPrefixes):
                    ns_to_render.append((n, v))
                elif not inclusive:
                    ns_unused_inherited[n] = v
        ns_to_render.sort(_sorter_ns)
        for n, v in ns_to_render:
            self._do_attr(n, v)
            ns_rendered[n] = v
        if not inclusive or _in_subset(self.subset, node.parentNode):
            other_attrs.extend(xml_attrs_local.values())
        else:
            other_attrs.extend(xml_attrs.values())
        other_attrs.sort(_sorter)
        for a in other_attrs:
            self._do_attr(a.nodeName, a.value)
        W('>')
    state, self.state = (self.state, (ns_local, ns_rendered, xml_attrs, ns_unused_inherited))
    for c in _children(node):
        _implementation.handlers[c.nodeType](self, c)
    self.state = state
    if name:
        W('</%s>' % name)