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
class NoDefaultALPNParam(Param):

    @classmethod
    def emptiness(cls):
        return Emptiness.ALWAYS

    @classmethod
    def from_value(cls, value):
        if value is None or value == '':
            return None
        else:
            raise ValueError('no-default-alpn with non-empty value')

    def to_text(self):
        raise NotImplementedError

    @classmethod
    def from_wire_parser(cls, parser, origin=None):
        if parser.remaining() != 0:
            raise dns.exception.FormError
        return None

    def to_wire(self, file, origin=None):
        raise NotImplementedError