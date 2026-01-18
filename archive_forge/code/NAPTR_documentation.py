import struct
import dns.exception
import dns.name
import dns.rdata
from dns._compat import xrange, text_type
NAPTR record

    @ivar order: order
    @type order: int
    @ivar preference: preference
    @type preference: int
    @ivar flags: flags
    @type flags: string
    @ivar service: service
    @type service: string
    @ivar regexp: regular expression
    @type regexp: string
    @ivar replacement: replacement name
    @type replacement: dns.name.Name object
    @see: RFC 3403