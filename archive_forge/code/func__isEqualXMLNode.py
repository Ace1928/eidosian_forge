import re
from fractions import Fraction
import logging
import math
import warnings
import xml.dom.minidom
from base64 import b64decode, b64encode
from binascii import hexlify, unhexlify
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from re import compile, sub
from typing import (
from urllib.parse import urldefrag, urljoin, urlparse
from isodate import (
import rdflib
import rdflib.util
from rdflib.compat import long_type
def _isEqualXMLNode(node: Union[None, xml.dom.minidom.Attr, xml.dom.minidom.Comment, xml.dom.minidom.Document, xml.dom.minidom.DocumentFragment, xml.dom.minidom.DocumentType, xml.dom.minidom.Element, xml.dom.minidom.Entity, xml.dom.minidom.Notation, xml.dom.minidom.ProcessingInstruction, xml.dom.minidom.Text], other: Union[None, xml.dom.minidom.Attr, xml.dom.minidom.Comment, xml.dom.minidom.Document, xml.dom.minidom.DocumentFragment, xml.dom.minidom.DocumentType, xml.dom.minidom.Element, xml.dom.minidom.Entity, xml.dom.minidom.Notation, xml.dom.minidom.ProcessingInstruction, xml.dom.minidom.Text]) -> bool:
    from xml.dom.minidom import Node as XMLNode

    def recurse():
        if len(node.childNodes) != len(other.childNodes):
            return False
        for nc, oc in map(lambda x, y: (x, y), node.childNodes, other.childNodes):
            if not _isEqualXMLNode(nc, oc):
                return False
        return True
    if node is None or other is None:
        return False
    if node.nodeType != other.nodeType:
        return False
    if node.nodeType in [XMLNode.DOCUMENT_NODE, XMLNode.DOCUMENT_FRAGMENT_NODE]:
        return recurse()
    elif node.nodeType == XMLNode.ELEMENT_NODE:
        if TYPE_CHECKING:
            assert isinstance(node, xml.dom.minidom.Element)
            assert isinstance(other, xml.dom.minidom.Element)
        if not (node.tagName == other.tagName and node.namespaceURI == other.namespaceURI):
            return False
        n_keys = [k for k in node.attributes.keysNS() if k[0] != 'http://www.w3.org/2000/xmlns/']
        o_keys = [k for k in other.attributes.keysNS() if k[0] != 'http://www.w3.org/2000/xmlns/']
        if len(n_keys) != len(o_keys):
            return False
        for k in n_keys:
            if not (k in o_keys and node.getAttributeNS(k[0], k[1]) == other.getAttributeNS(k[0], k[1])):
                return False
        return recurse()
    elif node.nodeType in [XMLNode.TEXT_NODE, XMLNode.COMMENT_NODE, XMLNode.CDATA_SECTION_NODE, XMLNode.NOTATION_NODE]:
        if TYPE_CHECKING:
            assert isinstance(node, (xml.dom.minidom.Text, xml.dom.minidom.Comment, xml.dom.minidom.CDATASection, xml.dom.minidom.Notation))
            assert isinstance(other, (xml.dom.minidom.Text, xml.dom.minidom.Comment, xml.dom.minidom.CDATASection, xml.dom.minidom.Notation))
        return node.data == other.data
    elif node.nodeType == XMLNode.PROCESSING_INSTRUCTION_NODE:
        if TYPE_CHECKING:
            assert isinstance(node, xml.dom.minidom.ProcessingInstruction)
            assert isinstance(other, xml.dom.minidom.ProcessingInstruction)
        return node.data == other.data and node.target == other.target
    elif node.nodeType == XMLNode.ENTITY_NODE:
        return node.nodeValue == other.nodeValue
    elif node.nodeType == XMLNode.DOCUMENT_TYPE_NODE:
        if TYPE_CHECKING:
            assert isinstance(node, xml.dom.minidom.DocumentType)
            assert isinstance(other, xml.dom.minidom.DocumentType)
        return node.publicId == other.publicId and node.systemId == other.systemId
    else:
        raise Exception('I dont know how to compare XML Node type: %s' % node.nodeType)