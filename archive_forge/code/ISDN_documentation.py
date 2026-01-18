import struct
import dns.exception
import dns.rdata
import dns.tokenizer
from dns._compat import text_type
ISDN record

    @ivar address: the ISDN address
    @type address: string
    @ivar subaddress: the ISDN subaddress (or '' if not present)
    @type subaddress: string
    @see: RFC 1183