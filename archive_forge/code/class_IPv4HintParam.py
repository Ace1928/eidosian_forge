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
class IPv4HintParam(Param):

    def __init__(self, addresses):
        self.addresses = dns.rdata.Rdata._as_tuple(addresses, dns.rdata.Rdata._as_ipv4_address)

    @classmethod
    def from_value(cls, value):
        addresses = value.split(',')
        return cls(addresses)

    def to_text(self):
        return '"' + ','.join(self.addresses) + '"'

    @classmethod
    def from_wire_parser(cls, parser, origin=None):
        addresses = []
        while parser.remaining() > 0:
            ip = parser.get_bytes(4)
            addresses.append(dns.ipv4.inet_ntoa(ip))
        return cls(addresses)

    def to_wire(self, file, origin=None):
        for address in self.addresses:
            file.write(dns.ipv4.inet_aton(address))