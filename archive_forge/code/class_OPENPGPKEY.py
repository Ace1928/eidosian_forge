import base64
import dns.exception
import dns.rdata
import dns.tokenizer
class OPENPGPKEY(dns.rdata.Rdata):
    """OPENPGPKEY record

    @ivar key: the key
    @type key: bytes
    @see: RFC 7929
    """

    def __init__(self, rdclass, rdtype, key):
        super(OPENPGPKEY, self).__init__(rdclass, rdtype)
        self.key = key

    def to_text(self, origin=None, relativize=True, **kw):
        return dns.rdata._base64ify(self.key)

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True):
        chunks = []
        while 1:
            t = tok.get().unescape()
            if t.is_eol_or_eof():
                break
            if not t.is_identifier():
                raise dns.exception.SyntaxError
            chunks.append(t.value.encode())
        b64 = b''.join(chunks)
        key = base64.b64decode(b64)
        return cls(rdclass, rdtype, key)

    def to_wire(self, file, compress=None, origin=None):
        file.write(self.key)

    @classmethod
    def from_wire(cls, rdclass, rdtype, wire, current, rdlen, origin=None):
        key = wire[current:current + rdlen].unwrap()
        return cls(rdclass, rdtype, key)