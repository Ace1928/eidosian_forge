from __future__ import annotations
import re
import warnings
from io import BytesIO, StringIO
from incremental import Version, getVersionString
from twisted.python.compat import ioType
from twisted.python.util import InsensitiveDict
from twisted.web.sux import ParseError, XMLParser
def _gotStandalone(self, factory, data):
    parent = self._getparent()
    te = factory(data, parent)
    if parent:
        parent.appendChild(te)
    elif self.beExtremelyLenient:
        self.documents.append(te)