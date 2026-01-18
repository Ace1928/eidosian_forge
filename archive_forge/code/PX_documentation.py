import struct
import dns.exception
import dns.rdata
import dns.name
PX record.

    @ivar preference: the preference value
    @type preference: int
    @ivar map822: the map822 name
    @type map822: dns.name.Name object
    @ivar mapx400: the mapx400 name
    @type mapx400: dns.name.Name object
    @see: RFC 2163