import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def _set_prefix(self, prefix):
    nsuri = self.namespaceURI
    if prefix == 'xmlns':
        if nsuri and nsuri != XMLNS_NAMESPACE:
            raise xml.dom.NamespaceErr("illegal use of 'xmlns' prefix for the wrong namespace")
    self._prefix = prefix
    if prefix is None:
        newName = self.localName
    else:
        newName = '%s:%s' % (prefix, self.localName)
    if self.ownerElement:
        _clear_id_cache(self.ownerElement)
    self.name = newName