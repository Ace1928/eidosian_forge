import base64
import binascii
import string
import struct
import dns.exception
import dns.rdata
import dns.rdatatype
from dns._compat import xrange, text_type, PY3
NSEC3 record

    @ivar algorithm: the hash algorithm number
    @type algorithm: int
    @ivar flags: the flags
    @type flags: int
    @ivar iterations: the number of iterations
    @type iterations: int
    @ivar salt: the salt
    @type salt: string
    @ivar next: the next name hash
    @type next: string
    @ivar windows: the windowed bitmap list
    @type windows: list of (window number, string) tuples