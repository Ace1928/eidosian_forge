import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def _cmp(self, other):
    if self._attrs is getattr(other, '_attrs', None):
        return 0
    else:
        return (id(self) > id(other)) - (id(self) < id(other))