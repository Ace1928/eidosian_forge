import struct
import binascii
import dns.rdata
import dns.rdatatype
Base class for rdata that is like a DS record

    @ivar key_tag: the key tag
    @type key_tag: int
    @ivar algorithm: the algorithm
    @type algorithm: int
    @ivar digest_type: the digest type
    @type digest_type: int
    @ivar digest: the digest
    @type digest: int
    @see: draft-ietf-dnsext-delegation-signer-14.txt