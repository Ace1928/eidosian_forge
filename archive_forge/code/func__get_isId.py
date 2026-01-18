import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def _get_isId(self):
    if self._is_id:
        return True
    doc = self.ownerDocument
    elem = self.ownerElement
    if doc is None or elem is None:
        return False
    info = doc._get_elem_info(elem)
    if info is None:
        return False
    if self.namespaceURI:
        return info.isIdNS(self.namespaceURI, self.localName)
    else:
        return info.isId(self.nodeName)