from functools import reduce
import logging
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import packet_utils
from os_ken.lib.packet import stream_parser
from os_ken.lib.stringify import StringifyMixin
from os_ken.lib import type_desc
@LSA.register_type(OSPF_NETWORK_LSA)
class NetworkLSA(LSA):
    _PACK_STR = '!4s'
    _PACK_LEN = struct.calcsize(_PACK_STR)

    def __init__(self, ls_age=0, options=0, type_=OSPF_NETWORK_LSA, id_='0.0.0.0', adv_router='0.0.0.0', ls_seqnum=0, checksum=None, length=None, mask='0.0.0.0', routers=None):
        routers = routers if routers else []
        self.mask = mask
        self.routers = routers
        super(NetworkLSA, self).__init__(ls_age, options, type_, id_, adv_router, ls_seqnum, checksum, length)

    @classmethod
    def parser(cls, buf):
        if len(buf) < cls._PACK_LEN:
            raise stream_parser.StreamParser.TooSmallException('%d < %d' % (len(buf), cls._PACK_LEN))
        binmask = buf[:cls._PACK_LEN]
        mask, = struct.unpack_from(cls._PACK_STR, bytes(binmask))
        mask = addrconv.ipv4.bin_to_text(mask)
        buf = buf[cls._PACK_LEN:]
        routers = []
        while buf:
            binrouter = buf[:cls._PACK_LEN]
            router, = struct.unpack_from(cls._PACK_STR, bytes(binrouter))
            router = addrconv.ipv4.bin_to_text(router)
            routers.append(router)
            buf = buf[cls._PACK_LEN:]
        return {'mask': mask, 'routers': routers}

    def serialize_tail(self):
        mask = addrconv.ipv4.text_to_bin(self.mask)
        routers = [addrconv.ipv4.text_to_bin(router) for router in self.routers]
        return bytearray(struct.pack('!' + '4s' * (1 + len(routers)), mask, *routers))