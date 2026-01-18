import base64
import enum
import struct
import dns.enum
import dns.exception
import dns.immutable
import dns.ipv4
import dns.ipv6
import dns.name
import dns.rdata
import dns.rdtypes.util
import dns.renderer
import dns.tokenizer
import dns.wire
@dns.immutable.immutable
class ALPNParam(Param):

    def __init__(self, ids):
        self.ids = dns.rdata.Rdata._as_tuple(ids, lambda x: dns.rdata.Rdata._as_bytes(x, True, 255, False))

    @classmethod
    def from_value(cls, value):
        return cls(_split(_unescape(value)))

    def to_text(self):
        value = ','.join([_escapify(id) for id in self.ids])
        return '"' + dns.rdata._escapify(value.encode()) + '"'

    @classmethod
    def from_wire_parser(cls, parser, origin=None):
        ids = []
        while parser.remaining() > 0:
            id = parser.get_counted_bytes()
            ids.append(id)
        return cls(ids)

    def to_wire(self, file, origin=None):
        for id in self.ids:
            file.write(struct.pack('!B', len(id)))
            file.write(id)