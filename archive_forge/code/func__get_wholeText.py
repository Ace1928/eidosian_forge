import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def _get_wholeText(self):
    L = [self.data]
    n = self.previousSibling
    while n is not None:
        if n.nodeType in (Node.TEXT_NODE, Node.CDATA_SECTION_NODE):
            L.insert(0, n.data)
            n = n.previousSibling
        else:
            break
    n = self.nextSibling
    while n is not None:
        if n.nodeType in (Node.TEXT_NODE, Node.CDATA_SECTION_NODE):
            L.append(n.data)
            n = n.nextSibling
        else:
            break
    return ''.join(L)