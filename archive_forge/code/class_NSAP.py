import binascii
import dns.exception
import dns.rdata
import dns.tokenizer
class NSAP(dns.rdata.Rdata):
    """NSAP record.

    @ivar address: a NASP
    @type address: string
    @see: RFC 1706"""
    __slots__ = ['address']

    def __init__(self, rdclass, rdtype, address):
        super(NSAP, self).__init__(rdclass, rdtype)
        self.address = address

    def to_text(self, origin=None, relativize=True, **kw):
        return '0x%s' % binascii.hexlify(self.address).decode()

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True):
        address = tok.get_string()
        tok.get_eol()
        if address[0:2] != '0x':
            raise dns.exception.SyntaxError('string does not start with 0x')
        address = address[2:].replace('.', '')
        if len(address) % 2 != 0:
            raise dns.exception.SyntaxError('hexstring has odd length')
        address = binascii.unhexlify(address.encode())
        return cls(rdclass, rdtype, address)

    def to_wire(self, file, compress=None, origin=None):
        file.write(self.address)

    @classmethod
    def from_wire(cls, rdclass, rdtype, wire, current, rdlen, origin=None):
        address = wire[current:current + rdlen].unwrap()
        return cls(rdclass, rdtype, address)