import code
import sys
import tokenize
from io import BytesIO
from traceback import format_exception
from types import TracebackType
from typing import Type
from twisted.conch import recvline
from twisted.internet import defer
from twisted.python.compat import _get_async_param
from twisted.python.htmlizer import TokenPrinter
from twisted.python.monkey import MonkeyPatcher
class VT102Writer:
    """
    Colorizer for Python tokens.

    A series of tokens are written to instances of this object.  Each is
    colored in a particular way.  The final line of the result of this is
    generally added to the output.
    """
    typeToColor = {'identifier': b'\x1b[31m', 'keyword': b'\x1b[32m', 'parameter': b'\x1b[33m', 'variable': b'\x1b[1;33m', 'string': b'\x1b[35m', 'number': b'\x1b[36m', 'op': b'\x1b[37m'}
    normalColor = b'\x1b[0m'

    def __init__(self):
        self.written = []

    def color(self, type):
        r = self.typeToColor.get(type, b'')
        return r

    def write(self, token, type=None):
        if token and token != b'\r':
            c = self.color(type)
            if c:
                self.written.append(c)
            self.written.append(token)
            if c:
                self.written.append(self.normalColor)

    def __bytes__(self):
        s = b''.join(self.written)
        return s.strip(b'\n').splitlines()[-1]
    if bytes == str:
        __str__ = __bytes__