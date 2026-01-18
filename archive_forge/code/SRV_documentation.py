import struct
import dns.exception
import dns.rdata
import dns.name
SRV record

    @ivar priority: the priority
    @type priority: int
    @ivar weight: the weight
    @type weight: int
    @ivar port: the port of the service
    @type port: int
    @ivar target: the target host
    @type target: dns.name.Name object
    @see: RFC 2782