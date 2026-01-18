import struct
import binascii
import dns.rdata
import dns.rdatatype
TLSA record

    @ivar usage: The certificate usage
    @type usage: int
    @ivar selector: The selector field
    @type selector: int
    @ivar mtype: The 'matching type' field
    @type mtype: int
    @ivar cert: The 'Certificate Association Data' field
    @type cert: string
    @see: RFC 6698