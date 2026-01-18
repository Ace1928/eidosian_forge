import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
class ElementInfo(object):
    """Object that represents content-model information for an element.

    This implementation is not expected to be used in practice; DOM
    builders should provide implementations which do the right thing
    using information available to it.

    """
    __slots__ = ('tagName',)

    def __init__(self, name):
        self.tagName = name

    def getAttributeType(self, aname):
        return _no_type

    def getAttributeTypeNS(self, namespaceURI, localName):
        return _no_type

    def isElementContent(self):
        return False

    def isEmpty(self):
        """Returns true iff this element is declared to have an EMPTY
        content model."""
        return False

    def isId(self, aname):
        """Returns true iff the named attribute is a DTD-style ID."""
        return False

    def isIdNS(self, namespaceURI, localName):
        """Returns true iff the identified attribute is a DTD-style ID."""
        return False

    def __getstate__(self):
        return self.tagName

    def __setstate__(self, state):
        self.tagName = state