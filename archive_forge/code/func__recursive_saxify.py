from xml.sax.handler import ContentHandler
from lxml import etree
from lxml.etree import ElementTree, SubElement
from lxml.etree import Comment, ProcessingInstruction
def _recursive_saxify(self, element, parent_nsmap):
    content_handler = self._content_handler
    tag = element.tag
    if tag is Comment or tag is ProcessingInstruction:
        if tag is ProcessingInstruction:
            content_handler.processingInstruction(element.target, element.text)
        tail = element.tail
        if tail:
            content_handler.characters(tail)
        return
    element_nsmap = element.nsmap
    new_prefixes = []
    if element_nsmap != parent_nsmap:
        for prefix, ns_uri in element_nsmap.items():
            if parent_nsmap.get(prefix) != ns_uri:
                new_prefixes.append((prefix, ns_uri))
    attribs = element.items()
    if attribs:
        attr_values = {}
        attr_qnames = {}
        for attr_ns_name, value in attribs:
            attr_ns_tuple = _getNsTag(attr_ns_name)
            attr_values[attr_ns_tuple] = value
            attr_qnames[attr_ns_tuple] = self._build_qname(attr_ns_tuple[0], attr_ns_tuple[1], element_nsmap, preferred_prefix=None, is_attribute=True)
        sax_attributes = self._attr_class(attr_values, attr_qnames)
    else:
        sax_attributes = self._empty_attributes
    ns_uri, local_name = _getNsTag(tag)
    qname = self._build_qname(ns_uri, local_name, element_nsmap, element.prefix, is_attribute=False)
    for prefix, uri in new_prefixes:
        content_handler.startPrefixMapping(prefix, uri)
    content_handler.startElementNS((ns_uri, local_name), qname, sax_attributes)
    text = element.text
    if text:
        content_handler.characters(text)
    for child in element:
        self._recursive_saxify(child, element_nsmap)
    content_handler.endElementNS((ns_uri, local_name), qname)
    for prefix, uri in new_prefixes:
        content_handler.endPrefixMapping(prefix)
    tail = element.tail
    if tail:
        content_handler.characters(tail)