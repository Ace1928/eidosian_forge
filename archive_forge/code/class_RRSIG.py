import base64
import calendar
import struct
import time
import dns.dnssec
import dns.exception
import dns.rdata
import dns.rdatatype
class RRSIG(dns.rdata.Rdata):
    """RRSIG record

    @ivar type_covered: the rdata type this signature covers
    @type type_covered: int
    @ivar algorithm: the algorithm used for the sig
    @type algorithm: int
    @ivar labels: number of labels
    @type labels: int
    @ivar original_ttl: the original TTL
    @type original_ttl: long
    @ivar expiration: signature expiration time
    @type expiration: long
    @ivar inception: signature inception time
    @type inception: long
    @ivar key_tag: the key tag
    @type key_tag: int
    @ivar signer: the signer
    @type signer: dns.name.Name object
    @ivar signature: the signature
    @type signature: string"""
    __slots__ = ['type_covered', 'algorithm', 'labels', 'original_ttl', 'expiration', 'inception', 'key_tag', 'signer', 'signature']

    def __init__(self, rdclass, rdtype, type_covered, algorithm, labels, original_ttl, expiration, inception, key_tag, signer, signature):
        super(RRSIG, self).__init__(rdclass, rdtype)
        self.type_covered = type_covered
        self.algorithm = algorithm
        self.labels = labels
        self.original_ttl = original_ttl
        self.expiration = expiration
        self.inception = inception
        self.key_tag = key_tag
        self.signer = signer
        self.signature = signature

    def covers(self):
        return self.type_covered

    def to_text(self, origin=None, relativize=True, **kw):
        return '%s %d %d %d %s %s %d %s %s' % (dns.rdatatype.to_text(self.type_covered), self.algorithm, self.labels, self.original_ttl, posixtime_to_sigtime(self.expiration), posixtime_to_sigtime(self.inception), self.key_tag, self.signer.choose_relativity(origin, relativize), dns.rdata._base64ify(self.signature))

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True):
        type_covered = dns.rdatatype.from_text(tok.get_string())
        algorithm = dns.dnssec.algorithm_from_text(tok.get_string())
        labels = tok.get_int()
        original_ttl = tok.get_ttl()
        expiration = sigtime_to_posixtime(tok.get_string())
        inception = sigtime_to_posixtime(tok.get_string())
        key_tag = tok.get_int()
        signer = tok.get_name()
        signer = signer.choose_relativity(origin, relativize)
        chunks = []
        while 1:
            t = tok.get().unescape()
            if t.is_eol_or_eof():
                break
            if not t.is_identifier():
                raise dns.exception.SyntaxError
            chunks.append(t.value.encode())
        b64 = b''.join(chunks)
        signature = base64.b64decode(b64)
        return cls(rdclass, rdtype, type_covered, algorithm, labels, original_ttl, expiration, inception, key_tag, signer, signature)

    def to_wire(self, file, compress=None, origin=None):
        header = struct.pack('!HBBIIIH', self.type_covered, self.algorithm, self.labels, self.original_ttl, self.expiration, self.inception, self.key_tag)
        file.write(header)
        self.signer.to_wire(file, None, origin)
        file.write(self.signature)

    @classmethod
    def from_wire(cls, rdclass, rdtype, wire, current, rdlen, origin=None):
        header = struct.unpack('!HBBIIIH', wire[current:current + 18])
        current += 18
        rdlen -= 18
        signer, cused = dns.name.from_wire(wire[:current + rdlen], current)
        current += cused
        rdlen -= cused
        if origin is not None:
            signer = signer.relativize(origin)
        signature = wire[current:current + rdlen].unwrap()
        return cls(rdclass, rdtype, header[0], header[1], header[2], header[3], header[4], header[5], header[6], signer, signature)

    def choose_relativity(self, origin=None, relativize=True):
        self.signer = self.signer.choose_relativity(origin, relativize)