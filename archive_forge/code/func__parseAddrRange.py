from paste.util import intset
import socket
def _parseAddrRange(self, addr):
    naddr, naddrlen = _parseAddr(addr)
    naddr1 = naddr << (4 - naddrlen) * 8
    naddr2 = (naddr << (4 - naddrlen) * 8) + (1 << (4 - naddrlen) * 8) - 1
    return (naddr1, naddr2)