import struct
import binascii
import dns.exception
import dns.rdata
from dns._compat import text_type
NSEC3PARAM record

    @ivar algorithm: the hash algorithm number
    @type algorithm: int
    @ivar flags: the flags
    @type flags: int
    @ivar iterations: the number of iterations
    @type iterations: int
    @ivar salt: the salt
    @type salt: string