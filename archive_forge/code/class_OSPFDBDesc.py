from functools import reduce
import logging
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import packet_utils
from os_ken.lib.packet import stream_parser
from os_ken.lib.stringify import StringifyMixin
from os_ken.lib import type_desc
@OSPFMessage.register_type(OSPF_MSG_DB_DESC)
class OSPFDBDesc(OSPFMessage):
    _PACK_STR = '!HBBI'
    _PACK_LEN = struct.calcsize(_PACK_STR)
    _MIN_LEN = OSPFMessage._HDR_LEN + _PACK_LEN

    def __init__(self, length=None, router_id='0.0.0.0', area_id='0.0.0.0', au_type=1, authentication=0, checksum=None, version=_VERSION, mtu=1500, options=0, i_flag=0, m_flag=0, ms_flag=0, sequence_number=0, lsa_headers=None):
        lsa_headers = lsa_headers if lsa_headers else []
        super(OSPFDBDesc, self).__init__(OSPF_MSG_DB_DESC, length, router_id, area_id, au_type, authentication, checksum, version)
        self.mtu = mtu
        self.options = options
        self.i_flag = i_flag
        self.m_flag = m_flag
        self.ms_flag = ms_flag
        self.sequence_number = sequence_number
        self.lsa_headers = lsa_headers

    @classmethod
    def parser(cls, buf):
        mtu, options, flags, sequence_number = struct.unpack_from(cls._PACK_STR, bytes(buf))
        i_flag = flags >> 2 & 1
        m_flag = flags >> 1 & 1
        ms_flag = flags & 1
        lsahdrs = []
        buf = buf[cls._PACK_LEN:]
        while buf:
            kwargs, buf = LSAHeader.parser(buf)
            lsahdrs.append(LSAHeader(**kwargs))
        return {'mtu': mtu, 'options': options, 'i_flag': i_flag, 'm_flag': m_flag, 'ms_flag': ms_flag, 'sequence_number': sequence_number, 'lsa_headers': lsahdrs}

    def serialize_tail(self):
        flags = (self.i_flag & 1) << 2 ^ (self.m_flag & 1) << 1 ^ self.ms_flag & 1
        head = bytearray(struct.pack(self._PACK_STR, self.mtu, self.options, flags, self.sequence_number))
        try:
            return head + reduce(lambda a, b: a + b, (hdr.serialize() for hdr in self.lsa_headers))
        except TypeError:
            return head