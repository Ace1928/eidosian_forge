from zope.interface import Interface, implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.python.compat import iterbytes, networkString
def _handleLowFunctionControlSequence(self, ch):
    functionKeys = {b'P': self.F1, b'Q': self.F2, b'R': self.F3, b'S': self.F4}
    keyID = functionKeys.get(ch)
    if keyID is not None:
        self.terminalProtocol.keystrokeReceived(keyID, None)
    else:
        self.terminalProtocol.unhandledControlSequence(b'\x1b[O' + ch)