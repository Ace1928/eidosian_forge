import struct
import dns.exception
import dns.rdata
import dns.tokenizer
from dns._compat import text_type
class X25(dns.rdata.Rdata):
    """X25 record

    @ivar address: the PSDN address
    @type address: string
    @see: RFC 1183"""
    __slots__ = ['address']

    def __init__(self, rdclass, rdtype, address):
        super(X25, self).__init__(rdclass, rdtype)
        if isinstance(address, text_type):
            self.address = address.encode()
        else:
            self.address = address

    def to_text(self, origin=None, relativize=True, **kw):
        return '"%s"' % dns.rdata._escapify(self.address)

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True):
        address = tok.get_string()
        tok.get_eol()
        return cls(rdclass, rdtype, address)

    def to_wire(self, file, compress=None, origin=None):
        l = len(self.address)
        assert l < 256
        file.write(struct.pack('!B', l))
        file.write(self.address)

    @classmethod
    def from_wire(cls, rdclass, rdtype, wire, current, rdlen, origin=None):
        l = wire[current]
        current += 1
        rdlen -= 1
        if l != rdlen:
            raise dns.exception.FormError
        address = wire[current:current + l].unwrap()
        return cls(rdclass, rdtype, address)