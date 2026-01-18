from xml.sax.handler import ContentHandler
from lxml import etree
from lxml.etree import ElementTree, SubElement
from lxml.etree import Comment, ProcessingInstruction
class SaxError(etree.LxmlError):
    """General SAX error.
    """