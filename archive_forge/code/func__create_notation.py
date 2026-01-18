import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def _create_notation(self, name, publicId, systemId):
    n = Notation(name, publicId, systemId)
    n.ownerDocument = self
    return n