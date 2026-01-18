from paste.util import intset
import socket
def _parseRange(self, addr1, addr2):
    naddr1, naddr1len = _parseAddr(addr1)
    naddr2, naddr2len = _parseAddr(addr2)
    if naddr2len < naddr1len:
        naddr2 += naddr1 & (1 << (naddr1len - naddr2len) * 8) - 1 << naddr2len * 8
        naddr2len = naddr1len
    elif naddr2len > naddr1len:
        raise ValueError('Range has more dots than address.')
    naddr1 <<= (4 - naddr1len) * 8
    naddr2 <<= (4 - naddr2len) * 8
    naddr2 += (1 << (4 - naddr2len) * 8) - 1
    return (naddr1, naddr2)