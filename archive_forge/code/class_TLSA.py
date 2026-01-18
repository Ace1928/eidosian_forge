import struct
import binascii
import dns.rdata
import dns.rdatatype
class TLSA(dns.rdata.Rdata):
    """TLSA record

    @ivar usage: The certificate usage
    @type usage: int
    @ivar selector: The selector field
    @type selector: int
    @ivar mtype: The 'matching type' field
    @type mtype: int
    @ivar cert: The 'Certificate Association Data' field
    @type cert: string
    @see: RFC 6698"""
    __slots__ = ['usage', 'selector', 'mtype', 'cert']

    def __init__(self, rdclass, rdtype, usage, selector, mtype, cert):
        super(TLSA, self).__init__(rdclass, rdtype)
        self.usage = usage
        self.selector = selector
        self.mtype = mtype
        self.cert = cert

    def to_text(self, origin=None, relativize=True, **kw):
        return '%d %d %d %s' % (self.usage, self.selector, self.mtype, dns.rdata._hexify(self.cert, chunksize=128))

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True):
        usage = tok.get_uint8()
        selector = tok.get_uint8()
        mtype = tok.get_uint8()
        cert_chunks = []
        while 1:
            t = tok.get().unescape()
            if t.is_eol_or_eof():
                break
            if not t.is_identifier():
                raise dns.exception.SyntaxError
            cert_chunks.append(t.value.encode())
        cert = b''.join(cert_chunks)
        cert = binascii.unhexlify(cert)
        return cls(rdclass, rdtype, usage, selector, mtype, cert)

    def to_wire(self, file, compress=None, origin=None):
        header = struct.pack('!BBB', self.usage, self.selector, self.mtype)
        file.write(header)
        file.write(self.cert)

    @classmethod
    def from_wire(cls, rdclass, rdtype, wire, current, rdlen, origin=None):
        header = struct.unpack('!BBB', wire[current:current + 3])
        current += 3
        rdlen -= 3
        cert = wire[current:current + rdlen].unwrap()
        return cls(rdclass, rdtype, header[0], header[1], header[2], cert)