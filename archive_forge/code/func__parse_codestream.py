from __future__ import annotations
import io
import os
import struct
from . import Image, ImageFile, _binary
def _parse_codestream(fp):
    """Parse the JPEG 2000 codestream to extract the size and component
    count from the SIZ marker segment, returning a PIL (size, mode) tuple."""
    hdr = fp.read(2)
    lsiz = _binary.i16be(hdr)
    siz = hdr + fp.read(lsiz - 2)
    lsiz, rsiz, xsiz, ysiz, xosiz, yosiz, _, _, _, _, csiz = struct.unpack_from('>HHIIIIIIIIH', siz)
    ssiz = [None] * csiz
    xrsiz = [None] * csiz
    yrsiz = [None] * csiz
    for i in range(csiz):
        ssiz[i], xrsiz[i], yrsiz[i] = struct.unpack_from('>BBB', siz, 36 + 3 * i)
    size = (xsiz - xosiz, ysiz - yosiz)
    if csiz == 1:
        if yrsiz[0] & 127 > 8:
            mode = 'I;16'
        else:
            mode = 'L'
    elif csiz == 2:
        mode = 'LA'
    elif csiz == 3:
        mode = 'RGB'
    elif csiz == 4:
        mode = 'RGBA'
    else:
        mode = None
    return (size, mode)