from functools import reduce
import logging
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import packet_utils
from os_ken.lib.packet import stream_parser
from os_ken.lib.stringify import StringifyMixin
from os_ken.lib import type_desc
class LSA(type_desc.TypeDisp, StringifyMixin):

    def __init__(self, ls_age=0, options=0, type_=OSPF_UNKNOWN_LSA, id_='0.0.0.0', adv_router='0.0.0.0', ls_seqnum=0, checksum=0, length=0, opaque_type=OSPF_OPAQUE_TYPE_UNKNOWN, opaque_id=0):
        if type_ < OSPF_OPAQUE_LINK_LSA:
            self.header = LSAHeader(ls_age=ls_age, options=options, type_=type_, id_=id_, adv_router=adv_router, ls_seqnum=ls_seqnum)
        else:
            self.header = LSAHeader(ls_age=ls_age, options=options, type_=type_, adv_router=adv_router, ls_seqnum=ls_seqnum, opaque_type=opaque_type, opaque_id=opaque_id)
        if not (checksum or length):
            tail = self.serialize_tail()
            length = self.header._HDR_LEN + len(tail)
        if not checksum:
            head = self.header.serialize()
            checksum = packet_utils.fletcher_checksum(head[2:], 14)
        self.header.length = length
        self.header.checksum = checksum

    @classmethod
    def parser(cls, buf):
        hdr, rest = LSAHeader.parser(buf)
        if len(buf) < hdr['length']:
            raise stream_parser.StreamParser.TooSmallException('%d < %d' % (len(buf), hdr['length']))
        csum = packet_utils.fletcher_checksum(buf[2:hdr['length']], 14)
        if csum != hdr['checksum']:
            raise InvalidChecksum('header has %d, but calculated value is %d' % (hdr['checksum'], csum))
        subcls = cls._lookup_type(hdr['type_'])
        body = rest[:hdr['length'] - LSAHeader._HDR_LEN]
        rest = rest[hdr['length'] - LSAHeader._HDR_LEN:]
        if issubclass(subcls, OpaqueLSA):
            kwargs = subcls.parser(body, hdr['opaque_type'])
        else:
            kwargs = subcls.parser(body)
        kwargs.update(hdr)
        return (subcls(**kwargs), subcls, rest)

    def serialize(self):
        tail = self.serialize_tail()
        self.header.length = self.header._HDR_LEN + len(tail)
        head = self.header.serialize()
        csum = packet_utils.fletcher_checksum(head[2:] + tail, 14)
        self.header.checksum = csum
        struct.pack_into('!H', head, 16, csum)
        return head + tail

    def serialize_tail(self):
        return b''