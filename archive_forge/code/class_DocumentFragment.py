import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
class DocumentFragment(Node):
    nodeType = Node.DOCUMENT_FRAGMENT_NODE
    nodeName = '#document-fragment'
    nodeValue = None
    attributes = None
    parentNode = None
    _child_node_types = (Node.ELEMENT_NODE, Node.TEXT_NODE, Node.CDATA_SECTION_NODE, Node.ENTITY_REFERENCE_NODE, Node.PROCESSING_INSTRUCTION_NODE, Node.COMMENT_NODE, Node.NOTATION_NODE)

    def __init__(self):
        self.childNodes = NodeList()