from __future__ import annotations
import re
import warnings
from io import BytesIO, StringIO
from incremental import Version, getVersionString
from twisted.python.compat import ioType
from twisted.python.util import InsensitiveDict
from twisted.web.sux import ParseError, XMLParser
class lmx:
    """
    Easy creation of XML.
    """

    def __init__(self, node='div'):
        if isinstance(node, str):
            node = Element(node)
        self.node = node

    def __getattr__(self, name):
        if name[0] == '_':
            raise AttributeError('no private attrs')
        return lambda **kw: self.add(name, **kw)

    def __setitem__(self, key, val):
        self.node.setAttribute(key, val)

    def __getitem__(self, key):
        return self.node.getAttribute(key)

    def text(self, txt, raw=0):
        nn = Text(txt, raw=raw)
        self.node.appendChild(nn)
        return self

    def add(self, tagName, **kw):
        newNode = Element(tagName, caseInsensitive=0, preserveCase=0)
        self.node.appendChild(newNode)
        xf = lmx(newNode)
        for k, v in kw.items():
            if k[0] == '_':
                k = k[1:]
            xf[k] = v
        return xf