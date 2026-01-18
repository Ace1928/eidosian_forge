import struct
import binascii
import dns.rdata
import dns.rdatatype
SSHFP record

    @ivar algorithm: the algorithm
    @type algorithm: int
    @ivar fp_type: the digest type
    @type fp_type: int
    @ivar fingerprint: the fingerprint
    @type fingerprint: string
    @see: draft-ietf-secsh-dns-05.txt