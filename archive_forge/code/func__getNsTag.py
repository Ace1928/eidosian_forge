from xml.sax.handler import ContentHandler
from lxml import etree
from lxml.etree import ElementTree, SubElement
from lxml.etree import Comment, ProcessingInstruction
def _getNsTag(tag):
    if tag[0] == '{':
        return tuple(tag[1:].split('}', 1))
    else:
        return (None, tag)