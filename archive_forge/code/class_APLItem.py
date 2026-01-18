import binascii
import codecs
import struct
import dns.exception
import dns.inet
import dns.rdata
import dns.tokenizer
from dns._compat import xrange, maybe_chr
class APLItem(object):
    """An APL list item.

    @ivar family: the address family (IANA address family registry)
    @type family: int
    @ivar negation: is this item negated?
    @type negation: bool
    @ivar address: the address
    @type address: string
    @ivar prefix: the prefix length
    @type prefix: int
    """
    __slots__ = ['family', 'negation', 'address', 'prefix']

    def __init__(self, family, negation, address, prefix):
        self.family = family
        self.negation = negation
        self.address = address
        self.prefix = prefix

    def __str__(self):
        if self.negation:
            return '!%d:%s/%s' % (self.family, self.address, self.prefix)
        else:
            return '%d:%s/%s' % (self.family, self.address, self.prefix)

    def to_wire(self, file):
        if self.family == 1:
            address = dns.inet.inet_pton(dns.inet.AF_INET, self.address)
        elif self.family == 2:
            address = dns.inet.inet_pton(dns.inet.AF_INET6, self.address)
        else:
            address = binascii.unhexlify(self.address)
        last = 0
        for i in xrange(len(address) - 1, -1, -1):
            if address[i] != maybe_chr(0):
                last = i + 1
                break
        address = address[0:last]
        l = len(address)
        assert l < 128
        if self.negation:
            l |= 128
        header = struct.pack('!HBB', self.family, self.prefix, l)
        file.write(header)
        file.write(address)