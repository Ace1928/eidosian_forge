from a string or file.
from xml.dom import xmlbuilder, minidom, Node
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE
from xml.parsers import expat
from xml.dom.minidom import _append_child, _set_attribute_node
from xml.dom.NodeFilter import NodeFilter
class Namespaces:
    """Mix-in class for builders; adds support for namespaces."""

    def _initNamespaces(self):
        self._ns_ordered_prefixes = []

    def createParser(self):
        """Create a new namespace-handling parser."""
        parser = expat.ParserCreate(namespace_separator=' ')
        parser.namespace_prefixes = True
        return parser

    def install(self, parser):
        """Insert the namespace-handlers onto the parser."""
        ExpatBuilder.install(self, parser)
        if self._options.namespace_declarations:
            parser.StartNamespaceDeclHandler = self.start_namespace_decl_handler

    def start_namespace_decl_handler(self, prefix, uri):
        """Push this namespace declaration on our storage."""
        self._ns_ordered_prefixes.append((prefix, uri))

    def start_element_handler(self, name, attributes):
        if ' ' in name:
            uri, localname, prefix, qname = _parse_ns_name(self, name)
        else:
            uri = EMPTY_NAMESPACE
            qname = name
            localname = None
            prefix = EMPTY_PREFIX
        node = minidom.Element(qname, uri, prefix, localname)
        node.ownerDocument = self.document
        _append_child(self.curNode, node)
        self.curNode = node
        if self._ns_ordered_prefixes:
            for prefix, uri in self._ns_ordered_prefixes:
                if prefix:
                    a = minidom.Attr(_intern(self, 'xmlns:' + prefix), XMLNS_NAMESPACE, prefix, 'xmlns')
                else:
                    a = minidom.Attr('xmlns', XMLNS_NAMESPACE, 'xmlns', EMPTY_PREFIX)
                a.value = uri
                a.ownerDocument = self.document
                _set_attribute_node(node, a)
            del self._ns_ordered_prefixes[:]
        if attributes:
            node._ensure_attributes()
            _attrs = node._attrs
            _attrsNS = node._attrsNS
            for i in range(0, len(attributes), 2):
                aname = attributes[i]
                value = attributes[i + 1]
                if ' ' in aname:
                    uri, localname, prefix, qname = _parse_ns_name(self, aname)
                    a = minidom.Attr(qname, uri, localname, prefix)
                    _attrs[qname] = a
                    _attrsNS[uri, localname] = a
                else:
                    a = minidom.Attr(aname, EMPTY_NAMESPACE, aname, EMPTY_PREFIX)
                    _attrs[aname] = a
                    _attrsNS[EMPTY_NAMESPACE, aname] = a
                a.ownerDocument = self.document
                a.value = value
                a.ownerElement = node
    if __debug__:

        def end_element_handler(self, name):
            curNode = self.curNode
            if ' ' in name:
                uri, localname, prefix, qname = _parse_ns_name(self, name)
                assert curNode.namespaceURI == uri and curNode.localName == localname and (curNode.prefix == prefix), 'element stack messed up! (namespace)'
            else:
                assert curNode.nodeName == name, 'element stack messed up - bad nodeName'
                assert curNode.namespaceURI == EMPTY_NAMESPACE, 'element stack messed up - bad namespaceURI'
            self.curNode = curNode.parentNode
            self._finish_end_element(curNode)