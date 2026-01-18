from zope.interface import Interface, implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.python.compat import iterbytes, networkString
def Z(self, proto, handler, buf):
    if buf == b'\x1b[':
        handler.keystrokeReceived(proto.TAB, proto.SHIFT)
    else:
        handler.unhandledControlSequence(buf + b'Z')