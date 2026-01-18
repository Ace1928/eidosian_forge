import struct
from os_ken import exception
from os_ken.lib import mac
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import inet
import logging
class NXMatch(object):

    def __init__(self, header):
        self.header = header

    @classmethod
    def parser(cls, buf, offset, match_len):
        if match_len < 4:
            raise exception.OFPMalformedMessage
        header, = struct.unpack_from(ofproto_v1_0.NXM_HEADER_PACK_STRING, buf, offset)
        instance = cls(header)
        payload_len = instance.length()
        if payload_len == 0 or match_len < payload_len + 4:
            raise exception.OFPMalformedMessage
        return instance

    def vendor(self):
        return self.header >> 16

    def field(self):
        return (self.header >> 9) % 127

    def type(self):
        return (self.header >> 9) % 8388607

    def hasmask(self):
        return self.header >> 8 & 1

    def length(self):
        return self.header & 255

    def show(self):
        return '%08x (vendor=%x, field=%x, hasmask=%x len=%x)' % (self.header, self.vendor(), self.field(), self.hasmask(), self.length())

    def put_header(self, buf, offset):
        msg_pack_into(ofproto_v1_0.NXM_HEADER_PACK_STRING, buf, offset, self.header)
        return struct.calcsize(ofproto_v1_0.NXM_HEADER_PACK_STRING)