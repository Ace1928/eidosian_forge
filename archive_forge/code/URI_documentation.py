import struct
import dns.exception
import dns.rdata
import dns.name
from dns._compat import text_type
URI record

    @ivar priority: the priority
    @type priority: int
    @ivar weight: the weight
    @type weight: int
    @ivar target: the target host
    @type target: dns.name.Name object
    @see: draft-faltstrom-uri-13