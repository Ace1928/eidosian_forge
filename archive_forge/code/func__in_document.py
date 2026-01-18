import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def _in_document(node):
    while node is not None:
        if node.nodeType == Node.DOCUMENT_NODE:
            return True
        node = node.parentNode
    return False