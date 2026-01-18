from functools import reduce
import logging
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import packet_utils
from os_ken.lib.packet import stream_parser
from os_ken.lib.stringify import StringifyMixin
from os_ken.lib import type_desc
@OSPFMessage.register_type(OSPF_MSG_LS_ACK)
class OSPFLSAck(OSPFMessage):
    _MIN_LEN = OSPFMessage._HDR_LEN

    def __init__(self, length=None, router_id='0.0.0.0', area_id='0.0.0.0', au_type=1, authentication=0, checksum=None, version=_VERSION, lsa_headers=None):
        lsa_headers = lsa_headers if lsa_headers else []
        super(OSPFLSAck, self).__init__(OSPF_MSG_LS_ACK, length, router_id, area_id, au_type, authentication, checksum, version)
        self.lsa_headers = lsa_headers

    @classmethod
    def parser(cls, buf):
        lsahdrs = []
        while buf:
            kwargs, buf = LSAHeader.parser(buf)
            lsahdrs.append(LSAHeader(**kwargs))
        return {'lsa_headers': lsahdrs}

    def serialize_tail(self):
        return reduce(lambda a, b: a + b, (hdr.serialize() for hdr in self.lsa_headers))