import struct
import dns.exception
import dns.rdata
import dns.tokenizer
from dns._compat import text_type
HINFO record

    @ivar cpu: the CPU type
    @type cpu: string
    @ivar os: the OS type
    @type os: string
    @see: RFC 1035