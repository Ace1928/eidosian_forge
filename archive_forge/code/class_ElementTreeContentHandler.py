from xml.sax.handler import ContentHandler
from lxml import etree
from lxml.etree import ElementTree, SubElement
from lxml.etree import Comment, ProcessingInstruction
class ElementTreeContentHandler(ContentHandler):
    """Build an lxml ElementTree from SAX events.
    """

    def __init__(self, makeelement=None):
        ContentHandler.__init__(self)
        self._root = None
        self._root_siblings = []
        self._element_stack = []
        self._default_ns = None
        self._ns_mapping = {None: [None]}
        self._new_mappings = {}
        if makeelement is None:
            makeelement = etree.Element
        self._makeelement = makeelement

    def _get_etree(self):
        """Contains the generated ElementTree after parsing is finished."""
        return ElementTree(self._root)
    etree = property(_get_etree, doc=_get_etree.__doc__)

    def setDocumentLocator(self, locator):
        pass

    def startDocument(self):
        pass

    def endDocument(self):
        pass

    def startPrefixMapping(self, prefix, uri):
        self._new_mappings[prefix] = uri
        try:
            self._ns_mapping[prefix].append(uri)
        except KeyError:
            self._ns_mapping[prefix] = [uri]
        if prefix is None:
            self._default_ns = uri

    def endPrefixMapping(self, prefix):
        ns_uri_list = self._ns_mapping[prefix]
        ns_uri_list.pop()
        if prefix is None:
            self._default_ns = ns_uri_list[-1]

    def _buildTag(self, ns_name_tuple):
        ns_uri, local_name = ns_name_tuple
        if ns_uri:
            el_tag = '{%s}%s' % ns_name_tuple
        elif self._default_ns:
            el_tag = '{%s}%s' % (self._default_ns, local_name)
        else:
            el_tag = local_name
        return el_tag

    def startElementNS(self, ns_name, qname, attributes=None):
        el_name = self._buildTag(ns_name)
        if attributes:
            attrs = {}
            try:
                iter_attributes = attributes.iteritems()
            except AttributeError:
                iter_attributes = attributes.items()
            for name_tuple, value in iter_attributes:
                if name_tuple[0]:
                    attr_name = '{%s}%s' % name_tuple
                else:
                    attr_name = name_tuple[1]
                attrs[attr_name] = value
        else:
            attrs = None
        element_stack = self._element_stack
        if self._root is None:
            element = self._root = self._makeelement(el_name, attrs, self._new_mappings)
            if self._root_siblings and hasattr(element, 'addprevious'):
                for sibling in self._root_siblings:
                    element.addprevious(sibling)
            del self._root_siblings[:]
        else:
            element = SubElement(element_stack[-1], el_name, attrs, self._new_mappings)
        element_stack.append(element)
        self._new_mappings.clear()

    def processingInstruction(self, target, data):
        pi = ProcessingInstruction(target, data)
        if self._root is None:
            self._root_siblings.append(pi)
        else:
            self._element_stack[-1].append(pi)

    def endElementNS(self, ns_name, qname):
        element = self._element_stack.pop()
        el_tag = self._buildTag(ns_name)
        if el_tag != element.tag:
            raise SaxError('Unexpected element closed: ' + el_tag)

    def startElement(self, name, attributes=None):
        if attributes:
            attributes = {(None, k): v for k, v in attributes.items()}
        self.startElementNS((None, name), name, attributes)

    def endElement(self, name):
        self.endElementNS((None, name), name)

    def characters(self, data):
        last_element = self._element_stack[-1]
        try:
            last_element = last_element[-1]
            last_element.tail = (last_element.tail or '') + data
        except IndexError:
            last_element.text = (last_element.text or '') + data
    ignorableWhitespace = characters