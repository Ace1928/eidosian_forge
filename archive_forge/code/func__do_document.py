import string
from xml.dom import Node
def _do_document(self, node):
    """_do_document(self, node) -> None
        Process a document node. documentOrder holds whether the document
        element has been encountered such that PIs/comments can be written
        as specified."""
    self.documentOrder = _LesserElement
    for child in node.childNodes:
        if child.nodeType == Node.ELEMENT_NODE:
            self.documentOrder = _Element
            self._do_element(child)
            self.documentOrder = _GreaterElement
        elif child.nodeType == Node.PROCESSING_INSTRUCTION_NODE:
            self._do_pi(child)
        elif child.nodeType == Node.COMMENT_NODE:
            self._do_comment(child)
        elif child.nodeType == Node.DOCUMENT_TYPE_NODE:
            pass
        else:
            raise TypeError(str(child))