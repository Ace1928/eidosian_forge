import platform
import socket
import struct
from os_ken.lib import addrconv
def _hdr(ss_len, af):
    if _HAVE_SS_LEN:
        return struct.pack(_HDR_FMT, ss_len, af)
    else:
        return struct.pack(_HDR_FMT, af)