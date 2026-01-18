import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
class Childless:
    """Mixin that makes childless-ness easy to implement and avoids
    the complexity of the Node methods that deal with children.
    """
    __slots__ = ()
    attributes = None
    childNodes = EmptyNodeList()
    firstChild = None
    lastChild = None

    def _get_firstChild(self):
        return None

    def _get_lastChild(self):
        return None

    def appendChild(self, node):
        raise xml.dom.HierarchyRequestErr(self.nodeName + ' nodes cannot have children')

    def hasChildNodes(self):
        return False

    def insertBefore(self, newChild, refChild):
        raise xml.dom.HierarchyRequestErr(self.nodeName + ' nodes do not have children')

    def removeChild(self, oldChild):
        raise xml.dom.NotFoundErr(self.nodeName + ' nodes do not have children')

    def normalize(self):
        pass

    def replaceChild(self, newChild, oldChild):
        raise xml.dom.HierarchyRequestErr(self.nodeName + ' nodes do not have children')