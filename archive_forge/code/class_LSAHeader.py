from functools import reduce
import logging
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import packet_utils
from os_ken.lib.packet import stream_parser
from os_ken.lib.stringify import StringifyMixin
from os_ken.lib import type_desc
class LSAHeader(StringifyMixin):
    _HDR_PACK_STR = '!HBB4s4sIHH'
    _HDR_LEN = struct.calcsize(_HDR_PACK_STR)

    def __init__(self, ls_age=0, options=0, type_=OSPF_UNKNOWN_LSA, id_='0.0.0.0', adv_router='0.0.0.0', ls_seqnum=0, checksum=0, length=0, opaque_type=OSPF_OPAQUE_TYPE_UNKNOWN, opaque_id=0):
        self.ls_age = ls_age
        self.options = options
        self.type_ = type_
        if self.type_ < OSPF_OPAQUE_LINK_LSA:
            self.id_ = id_
        else:
            self.opaque_type = opaque_type
            self.opaque_id = opaque_id
        self.adv_router = adv_router
        self.ls_seqnum = ls_seqnum
        self.checksum = checksum
        self.length = length

    @classmethod
    def parser(cls, buf):
        if len(buf) < cls._HDR_LEN:
            raise stream_parser.StreamParser.TooSmallException('%d < %d' % (len(buf), cls._HDR_LEN))
        ls_age, options, type_, id_, adv_router, ls_seqnum, checksum, length = struct.unpack_from(cls._HDR_PACK_STR, bytes(buf))
        adv_router = addrconv.ipv4.bin_to_text(adv_router)
        rest = buf[cls._HDR_LEN:]
        lsacls = LSA._lookup_type(type_)
        value = {'ls_age': ls_age, 'options': options, 'type_': type_, 'adv_router': adv_router, 'ls_seqnum': ls_seqnum, 'checksum': checksum, 'length': length}
        if issubclass(lsacls, OpaqueLSA):
            id_, = struct.unpack_from('!I', id_)
            value['opaque_type'] = (id_ & 4278190080) >> 24
            value['opaque_id'] = id_ & 16777215
        else:
            value['id_'] = addrconv.ipv4.bin_to_text(id_)
        return (value, rest)

    def serialize(self):
        if self.type_ < OSPF_OPAQUE_LINK_LSA:
            id_ = addrconv.ipv4.text_to_bin(self.id_)
        else:
            id_ = (self.opaque_type << 24) + self.opaque_id
            id_, = struct.unpack_from('4s', struct.pack('!I', id_))
        adv_router = addrconv.ipv4.text_to_bin(self.adv_router)
        return bytearray(struct.pack(self._HDR_PACK_STR, self.ls_age, self.options, self.type_, id_, adv_router, self.ls_seqnum, self.checksum, self.length))