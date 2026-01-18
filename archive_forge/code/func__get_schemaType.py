import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def _get_schemaType(self):
    doc = self.ownerDocument
    elem = self.ownerElement
    if doc is None or elem is None:
        return _no_type
    info = doc._get_elem_info(elem)
    if info is None:
        return _no_type
    if self.namespaceURI:
        return info.getAttributeTypeNS(self.namespaceURI, self.localName)
    else:
        return info.getAttributeType(self.nodeName)