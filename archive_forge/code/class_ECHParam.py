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
class ECHParam(Param):

    def __init__(self, ech):
        self.ech = dns.rdata.Rdata._as_bytes(ech, True)

    @classmethod
    def from_value(cls, value):
        if '\\' in value:
            raise ValueError('escape in ECH value')
        value = base64.b64decode(value.encode())
        return cls(value)

    def to_text(self):
        b64 = base64.b64encode(self.ech).decode('ascii')
        return f'"{b64}"'

    @classmethod
    def from_wire_parser(cls, parser, origin=None):
        value = parser.get_bytes(parser.remaining())
        return cls(value)

    def to_wire(self, file, origin=None):
        file.write(self.ech)