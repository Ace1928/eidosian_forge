from paste.util import intset
import socket
def _parseMask(self, addr, mask):
    naddr, naddrlen = _parseAddr(addr)
    naddr <<= (4 - naddrlen) * 8
    try:
        if not mask:
            masklen = 0
        else:
            masklen = int(mask)
        if not 0 <= masklen <= 32:
            raise ValueError
    except ValueError:
        try:
            mask = _parseAddr(mask, False)
        except ValueError:
            raise ValueError("Mask isn't parseable.")
        remaining = 0
        masklen = 0
        if not mask:
            masklen = 0
        else:
            while not mask & 1:
                remaining += 1
            while mask & 1:
                mask >>= 1
                masklen += 1
            if remaining + masklen != 32:
                raise ValueError("Mask isn't a proper host mask.")
    naddr1 = naddr & (1 << masklen) - 1 << 32 - masklen
    naddr2 = naddr1 + (1 << 32 - masklen) - 1
    return (naddr1, naddr2)