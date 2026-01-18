from zope.interface import Interface, implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.python.compat import iterbytes, networkString
def _handleHeightWidth(self, b):
    if b == b'3':
        self.terminal.doubleHeightLine(True)
    elif b == b'4':
        self.terminal.doubleHeightLine(False)
    elif b == b'5':
        self.terminal.singleWidthLine()
    elif b == b'6':
        self.terminal.doubleWidthLine()
    else:
        self.terminal.unhandledControlSequence(b'\x1b#' + b)