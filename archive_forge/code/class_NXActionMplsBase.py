import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionMplsBase(NXAction):
    _fmt_str = '!H4x'

    def __init__(self, ethertype, type_=None, len_=None, vendor=None, subtype=None):
        super(NXActionMplsBase, self).__init__()
        self.ethertype = ethertype

    @classmethod
    def parser(cls, buf):
        ethertype, = struct.unpack_from(cls._fmt_str, buf)
        return cls(ethertype)

    def serialize_body(self):
        data = bytearray()
        msg_pack_into(self._fmt_str, data, 0, self.ethertype)
        return data