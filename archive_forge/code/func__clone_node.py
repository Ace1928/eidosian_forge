import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def _clone_node(node, deep, newOwnerDocument):
    """
    Clone a node and give it the new owner document.
    Called by Node.cloneNode and Document.importNode
    """
    if node.ownerDocument.isSameNode(newOwnerDocument):
        operation = xml.dom.UserDataHandler.NODE_CLONED
    else:
        operation = xml.dom.UserDataHandler.NODE_IMPORTED
    if node.nodeType == Node.ELEMENT_NODE:
        clone = newOwnerDocument.createElementNS(node.namespaceURI, node.nodeName)
        for attr in node.attributes.values():
            clone.setAttributeNS(attr.namespaceURI, attr.nodeName, attr.value)
            a = clone.getAttributeNodeNS(attr.namespaceURI, attr.localName)
            a.specified = attr.specified
        if deep:
            for child in node.childNodes:
                c = _clone_node(child, deep, newOwnerDocument)
                clone.appendChild(c)
    elif node.nodeType == Node.DOCUMENT_FRAGMENT_NODE:
        clone = newOwnerDocument.createDocumentFragment()
        if deep:
            for child in node.childNodes:
                c = _clone_node(child, deep, newOwnerDocument)
                clone.appendChild(c)
    elif node.nodeType == Node.TEXT_NODE:
        clone = newOwnerDocument.createTextNode(node.data)
    elif node.nodeType == Node.CDATA_SECTION_NODE:
        clone = newOwnerDocument.createCDATASection(node.data)
    elif node.nodeType == Node.PROCESSING_INSTRUCTION_NODE:
        clone = newOwnerDocument.createProcessingInstruction(node.target, node.data)
    elif node.nodeType == Node.COMMENT_NODE:
        clone = newOwnerDocument.createComment(node.data)
    elif node.nodeType == Node.ATTRIBUTE_NODE:
        clone = newOwnerDocument.createAttributeNS(node.namespaceURI, node.nodeName)
        clone.specified = True
        clone.value = node.value
    elif node.nodeType == Node.DOCUMENT_TYPE_NODE:
        assert node.ownerDocument is not newOwnerDocument
        operation = xml.dom.UserDataHandler.NODE_IMPORTED
        clone = newOwnerDocument.implementation.createDocumentType(node.name, node.publicId, node.systemId)
        clone.ownerDocument = newOwnerDocument
        if deep:
            clone.entities._seq = []
            clone.notations._seq = []
            for n in node.notations._seq:
                notation = Notation(n.nodeName, n.publicId, n.systemId)
                notation.ownerDocument = newOwnerDocument
                clone.notations._seq.append(notation)
                if hasattr(n, '_call_user_data_handler'):
                    n._call_user_data_handler(operation, n, notation)
            for e in node.entities._seq:
                entity = Entity(e.nodeName, e.publicId, e.systemId, e.notationName)
                entity.actualEncoding = e.actualEncoding
                entity.encoding = e.encoding
                entity.version = e.version
                entity.ownerDocument = newOwnerDocument
                clone.entities._seq.append(entity)
                if hasattr(e, '_call_user_data_handler'):
                    e._call_user_data_handler(operation, e, entity)
    else:
        raise xml.dom.NotSupportedErr('Cannot clone node %s' % repr(node))
    if hasattr(node, '_call_user_data_handler'):
        node._call_user_data_handler(operation, node, clone)
    return clone