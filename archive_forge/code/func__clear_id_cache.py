import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def _clear_id_cache(node):
    if node.nodeType == Node.DOCUMENT_NODE:
        node._id_cache.clear()
        node._id_search_stack = None
    elif _in_document(node):
        node.ownerDocument._id_cache.clear()
        node.ownerDocument._id_search_stack = None