from functools import reduce
import logging
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import packet_utils
from os_ken.lib.packet import stream_parser
from os_ken.lib.stringify import StringifyMixin
from os_ken.lib import type_desc
@LSA.register_type(OSPF_SUMMARY_LSA)
class SummaryLSA(LSA):
    _PACK_STR = '!4sB3s'
    _PACK_LEN = struct.calcsize(_PACK_STR)

    def __init__(self, ls_age=0, options=0, type_=OSPF_SUMMARY_LSA, id_='0.0.0.0', adv_router='0.0.0.0', ls_seqnum=0, checksum=None, length=None, mask='0.0.0.0', tos=0, metric=0):
        self.mask = mask
        self.tos = tos
        self.metric = metric
        super(SummaryLSA, self).__init__(ls_age, options, type_, id_, adv_router, ls_seqnum, checksum, length)

    @classmethod
    def parser(cls, buf):
        if len(buf) < cls._PACK_LEN:
            raise stream_parser.StreamParser.TooSmallException('%d < %d' % (len(buf), cls._PACK_LEN))
        buf = buf[:cls._PACK_LEN]
        mask, tos, metric = struct.unpack_from(cls._PACK_STR, bytes(buf))
        mask = addrconv.ipv4.bin_to_text(mask)
        metric = type_desc.Int3.to_user(metric)
        return {'mask': mask, 'tos': tos, 'metric': metric}

    def serialize_tail(self):
        mask = addrconv.ipv4.text_to_bin(self.mask)
        metric = type_desc.Int3.from_user(self.metric)
        return bytearray(struct.pack(self._PACK_STR, mask, self.tos, metric))