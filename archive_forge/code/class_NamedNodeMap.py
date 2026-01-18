import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
class NamedNodeMap(object):
    """The attribute list is a transient interface to the underlying
    dictionaries.  Mutations here will change the underlying element's
    dictionary.

    Ordering is imposed artificially and does not reflect the order of
    attributes as found in an input document.
    """
    __slots__ = ('_attrs', '_attrsNS', '_ownerElement')

    def __init__(self, attrs, attrsNS, ownerElement):
        self._attrs = attrs
        self._attrsNS = attrsNS
        self._ownerElement = ownerElement

    def _get_length(self):
        return len(self._attrs)

    def item(self, index):
        try:
            return self[list(self._attrs.keys())[index]]
        except IndexError:
            return None

    def items(self):
        L = []
        for node in self._attrs.values():
            L.append((node.nodeName, node.value))
        return L

    def itemsNS(self):
        L = []
        for node in self._attrs.values():
            L.append(((node.namespaceURI, node.localName), node.value))
        return L

    def __contains__(self, key):
        if isinstance(key, str):
            return key in self._attrs
        else:
            return key in self._attrsNS

    def keys(self):
        return self._attrs.keys()

    def keysNS(self):
        return self._attrsNS.keys()

    def values(self):
        return self._attrs.values()

    def get(self, name, value=None):
        return self._attrs.get(name, value)
    __len__ = _get_length

    def _cmp(self, other):
        if self._attrs is getattr(other, '_attrs', None):
            return 0
        else:
            return (id(self) > id(other)) - (id(self) < id(other))

    def __eq__(self, other):
        return self._cmp(other) == 0

    def __ge__(self, other):
        return self._cmp(other) >= 0

    def __gt__(self, other):
        return self._cmp(other) > 0

    def __le__(self, other):
        return self._cmp(other) <= 0

    def __lt__(self, other):
        return self._cmp(other) < 0

    def __getitem__(self, attname_or_tuple):
        if isinstance(attname_or_tuple, tuple):
            return self._attrsNS[attname_or_tuple]
        else:
            return self._attrs[attname_or_tuple]

    def __setitem__(self, attname, value):
        if isinstance(value, str):
            try:
                node = self._attrs[attname]
            except KeyError:
                node = Attr(attname)
                node.ownerDocument = self._ownerElement.ownerDocument
                self.setNamedItem(node)
            node.value = value
        else:
            if not isinstance(value, Attr):
                raise TypeError('value must be a string or Attr object')
            node = value
            self.setNamedItem(node)

    def getNamedItem(self, name):
        try:
            return self._attrs[name]
        except KeyError:
            return None

    def getNamedItemNS(self, namespaceURI, localName):
        try:
            return self._attrsNS[namespaceURI, localName]
        except KeyError:
            return None

    def removeNamedItem(self, name):
        n = self.getNamedItem(name)
        if n is not None:
            _clear_id_cache(self._ownerElement)
            del self._attrs[n.nodeName]
            del self._attrsNS[n.namespaceURI, n.localName]
            if hasattr(n, 'ownerElement'):
                n.ownerElement = None
            return n
        else:
            raise xml.dom.NotFoundErr()

    def removeNamedItemNS(self, namespaceURI, localName):
        n = self.getNamedItemNS(namespaceURI, localName)
        if n is not None:
            _clear_id_cache(self._ownerElement)
            del self._attrsNS[n.namespaceURI, n.localName]
            del self._attrs[n.nodeName]
            if hasattr(n, 'ownerElement'):
                n.ownerElement = None
            return n
        else:
            raise xml.dom.NotFoundErr()

    def setNamedItem(self, node):
        if not isinstance(node, Attr):
            raise xml.dom.HierarchyRequestErr('%s cannot be child of %s' % (repr(node), repr(self)))
        old = self._attrs.get(node.name)
        if old:
            old.unlink()
        self._attrs[node.name] = node
        self._attrsNS[node.namespaceURI, node.localName] = node
        node.ownerElement = self._ownerElement
        _clear_id_cache(node.ownerElement)
        return old

    def setNamedItemNS(self, node):
        return self.setNamedItem(node)

    def __delitem__(self, attname_or_tuple):
        node = self[attname_or_tuple]
        _clear_id_cache(node.ownerElement)
        node.unlink()

    def __getstate__(self):
        return (self._attrs, self._attrsNS, self._ownerElement)

    def __setstate__(self, state):
        self._attrs, self._attrsNS, self._ownerElement = state