import struct
import base64
from os_ken.lib import addrconv
from os_ken.lib import mac
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib.packet import packet
from os_ken import exception
from os_ken import utils
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase
from os_ken.ofproto import ether
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import nx_actions
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_v1_3 as ofproto
import logging
@OFPTableFeatureProp.register_type(ofproto.OFPTFPT_NEXT_TABLES)
@OFPTableFeatureProp.register_type(ofproto.OFPTFPT_NEXT_TABLES_MISS)
class OFPTableFeaturePropNextTables(OFPTableFeatureProp):
    _TABLE_ID_PACK_STR = '!B'

    def __init__(self, type_=None, length=None, table_ids=None):
        table_ids = table_ids if table_ids else []
        super(OFPTableFeaturePropNextTables, self).__init__(type_, length)
        self.table_ids = table_ids

    @classmethod
    def parser(cls, buf):
        rest = cls.get_rest(buf)
        ids = []
        while rest:
            i, = struct.unpack_from(cls._TABLE_ID_PACK_STR, bytes(rest), 0)
            rest = rest[struct.calcsize(cls._TABLE_ID_PACK_STR):]
            ids.append(i)
        return cls(table_ids=ids)

    def serialize_body(self):
        bin_ids = bytearray()
        for i in self.table_ids:
            bin_id = bytearray()
            msg_pack_into(self._TABLE_ID_PACK_STR, bin_id, 0, i)
            bin_ids += bin_id
        return bin_ids