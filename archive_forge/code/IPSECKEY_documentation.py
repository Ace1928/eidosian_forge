import struct
import base64
import dns.exception
import dns.inet
import dns.name
IPSECKEY record

    @ivar precedence: the precedence for this key data
    @type precedence: int
    @ivar gateway_type: the gateway type
    @type gateway_type: int
    @ivar algorithm: the algorithm to use
    @type algorithm: int
    @ivar gateway: the public key
    @type gateway: None, IPv4 address, IPV6 address, or domain name
    @ivar key: the public key
    @type key: string
    @see: RFC 4025