from xml.sax.handler import ContentHandler
from lxml import etree
from lxml.etree import ElementTree, SubElement
from lxml.etree import Comment, ProcessingInstruction
def _get_etree(self):
    """Contains the generated ElementTree after parsing is finished."""
    return ElementTree(self._root)