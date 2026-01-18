from functools import reduce
import logging
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import packet_utils
from os_ken.lib.packet import stream_parser
from os_ken.lib.stringify import StringifyMixin
from os_ken.lib import type_desc
@LSA.register_type(OSPF_AS_EXTERNAL_LSA)
class ASExternalLSA(LSA):

    class ExternalNetwork(StringifyMixin):
        _PACK_STR = '!4sB3s4sI'
        _PACK_LEN = struct.calcsize(_PACK_STR)

        def __init__(self, mask='0.0.0.0', flags=0, metric=0, fwd_addr='0.0.0.0', tag=0):
            self.mask = mask
            self.flags = flags
            self.metric = metric
            self.fwd_addr = fwd_addr
            self.tag = tag

        @classmethod
        def parser(cls, buf):
            if len(buf) < cls._PACK_LEN:
                raise stream_parser.StreamParser.TooSmallException('%d < %d' % (len(buf), cls._PACK_LEN))
            ext_nw = buf[:cls._PACK_LEN]
            rest = buf[cls._PACK_LEN:]
            mask, flags, metric, fwd_addr, tag = struct.unpack_from(cls._PACK_STR, bytes(ext_nw))
            mask = addrconv.ipv4.bin_to_text(mask)
            metric = type_desc.Int3.to_user(metric)
            fwd_addr = addrconv.ipv4.bin_to_text(fwd_addr)
            return (cls(mask, flags, metric, fwd_addr, tag), rest)

        def serialize(self):
            mask = addrconv.ipv4.text_to_bin(self.mask)
            metric = type_desc.Int3.from_user(self.metric)
            fwd_addr = addrconv.ipv4.text_to_bin(self.fwd_addr)
            return bytearray(struct.pack(self._PACK_STR, mask, self.flags, metric, fwd_addr, self.tag))

    def __init__(self, ls_age=0, options=0, type_=OSPF_AS_EXTERNAL_LSA, id_='0.0.0.0', adv_router='0.0.0.0', ls_seqnum=0, checksum=None, length=None, extnws=None):
        extnws = extnws if extnws else []
        self.extnws = extnws
        super(ASExternalLSA, self).__init__(ls_age, options, type_, id_, adv_router, ls_seqnum, checksum, length)

    @classmethod
    def parser(cls, buf):
        extnws = []
        while buf:
            extnw, buf = cls.ExternalNetwork.parser(buf)
            extnws.append(extnw)
        return {'extnws': extnws}

    def serialize_tail(self):
        return reduce(lambda a, b: a + b, (extnw.serialize() for extnw in self.extnws))