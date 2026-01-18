import binascii
import codecs
import struct
import dns.exception
import dns.inet
import dns.rdata
import dns.tokenizer
from dns._compat import xrange, maybe_chr
class APL(dns.rdata.Rdata):
    """APL record.

    @ivar items: a list of APL items
    @type items: list of APL_Item
    @see: RFC 3123"""
    __slots__ = ['items']

    def __init__(self, rdclass, rdtype, items):
        super(APL, self).__init__(rdclass, rdtype)
        self.items = items

    def to_text(self, origin=None, relativize=True, **kw):
        return ' '.join(map(str, self.items))

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True):
        items = []
        while 1:
            token = tok.get().unescape()
            if token.is_eol_or_eof():
                break
            item = token.value
            if item[0] == '!':
                negation = True
                item = item[1:]
            else:
                negation = False
            family, rest = item.split(':', 1)
            family = int(family)
            address, prefix = rest.split('/', 1)
            prefix = int(prefix)
            item = APLItem(family, negation, address, prefix)
            items.append(item)
        return cls(rdclass, rdtype, items)

    def to_wire(self, file, compress=None, origin=None):
        for item in self.items:
            item.to_wire(file)

    @classmethod
    def from_wire(cls, rdclass, rdtype, wire, current, rdlen, origin=None):
        items = []
        while 1:
            if rdlen == 0:
                break
            if rdlen < 4:
                raise dns.exception.FormError
            header = struct.unpack('!HBB', wire[current:current + 4])
            afdlen = header[2]
            if afdlen > 127:
                negation = True
                afdlen -= 128
            else:
                negation = False
            current += 4
            rdlen -= 4
            if rdlen < afdlen:
                raise dns.exception.FormError
            address = wire[current:current + afdlen].unwrap()
            l = len(address)
            if header[0] == 1:
                if l < 4:
                    address += b'\x00' * (4 - l)
                address = dns.inet.inet_ntop(dns.inet.AF_INET, address)
            elif header[0] == 2:
                if l < 16:
                    address += b'\x00' * (16 - l)
                address = dns.inet.inet_ntop(dns.inet.AF_INET6, address)
            else:
                address = codecs.encode(address, 'hex_codec')
            current += afdlen
            rdlen -= afdlen
            item = APLItem(header[0], negation, address, header[1])
            items.append(item)
        return cls(rdclass, rdtype, items)