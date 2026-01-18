from a string or file.
from xml.dom import xmlbuilder, minidom, Node
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE
from xml.parsers import expat
from xml.dom.minidom import _append_child, _set_attribute_node
from xml.dom.NodeFilter import NodeFilter
def _handle_white_text_nodes(self, node, info):
    if self._options.whitespace_in_element_content or not info.isElementContent():
        return
    L = []
    for child in node.childNodes:
        if child.nodeType == TEXT_NODE and (not child.data.strip()):
            L.append(child)
    for child in L:
        node.removeChild(child)