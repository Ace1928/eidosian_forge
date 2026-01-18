from a string or file.
from xml.dom import xmlbuilder, minidom, Node
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE
from xml.parsers import expat
from xml.dom.minidom import _append_child, _set_attribute_node
from xml.dom.NodeFilter import NodeFilter
def _setup_subset(self, buffer):
    """Load the internal subset if there might be one."""
    if self.document.doctype:
        extractor = InternalSubsetExtractor()
        extractor.parseString(buffer)
        subset = extractor.getSubset()
        self.document.doctype.internalSubset = subset