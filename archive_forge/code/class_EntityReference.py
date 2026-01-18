from __future__ import annotations
import re
import warnings
from io import BytesIO, StringIO
from incremental import Version, getVersionString
from twisted.python.compat import ioType
from twisted.python.util import InsensitiveDict
from twisted.web.sux import ParseError, XMLParser
class EntityReference(Node):

    def __init__(self, eref, parentNode=None):
        Node.__init__(self, parentNode)
        self.eref = eref
        self.nodeValue = self.data = '&' + eref + ';'

    def isEqualToEntityReference(self, n):
        if not isinstance(n, EntityReference):
            return 0
        return self.eref == n.eref and self.nodeValue == n.nodeValue
    isEqualToNode = isEqualToEntityReference

    def writexml(self, stream, indent='', addindent='', newl='', strip=0, nsprefixes={}, namespace=''):
        w = _streamWriteWrapper(stream)
        w('' + self.nodeValue)

    def cloneNode(self, deep=0, parent=None):
        return EntityReference(self.eref, parent)