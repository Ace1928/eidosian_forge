from __future__ import annotations
import re
import warnings
from io import BytesIO, StringIO
from incremental import Version, getVersionString
from twisted.python.compat import ioType
from twisted.python.util import InsensitiveDict
from twisted.web.sux import ParseError, XMLParser
def _streamWriteWrapper(stream):
    if ioType(stream) == bytes:

        def w(s):
            if isinstance(s, str):
                s = s.encode('utf-8')
            stream.write(s)
    else:

        def w(s):
            if isinstance(s, bytes):
                s = s.decode('utf-8')
            stream.write(s)
    return w