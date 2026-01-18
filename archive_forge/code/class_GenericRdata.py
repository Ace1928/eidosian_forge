from io import BytesIO
import base64
import binascii
import dns.exception
import dns.name
import dns.rdataclass
import dns.rdatatype
import dns.tokenizer
import dns.wiredata
from ._compat import xrange, string_types, text_type
class GenericRdata(Rdata):
    """Generic Rdata Class

    This class is used for rdata types for which we have no better
    implementation.  It implements the DNS "unknown RRs" scheme.
    """
    __slots__ = ['data']

    def __init__(self, rdclass, rdtype, data):
        super(GenericRdata, self).__init__(rdclass, rdtype)
        self.data = data

    def to_text(self, origin=None, relativize=True, **kw):
        return '\\# %d ' % len(self.data) + _hexify(self.data)

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True):
        token = tok.get()
        if not token.is_identifier() or token.value != '\\#':
            raise dns.exception.SyntaxError('generic rdata does not start with \\#')
        length = tok.get_int()
        chunks = []
        while 1:
            token = tok.get()
            if token.is_eol_or_eof():
                break
            chunks.append(token.value.encode())
        hex = b''.join(chunks)
        data = binascii.unhexlify(hex)
        if len(data) != length:
            raise dns.exception.SyntaxError('generic rdata hex data has wrong length')
        return cls(rdclass, rdtype, data)

    def to_wire(self, file, compress=None, origin=None):
        file.write(self.data)

    @classmethod
    def from_wire(cls, rdclass, rdtype, wire, current, rdlen, origin=None):
        return cls(rdclass, rdtype, wire[current:current + rdlen])