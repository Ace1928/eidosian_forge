import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
@NXActionController2Prop.register_type(nicira_ext.NXAC2PT_MAX_LEN)
class NXActionController2PropMaxLen(NXActionController2Prop):
    _fmt_str = '!H2x'
    _arg_name = 'max_len'

    @classmethod
    def parser_prop(cls, buf, length):
        size = 4
        max_len, = struct.unpack_from(cls._fmt_str, buf, 0)
        return (max_len, size)

    @classmethod
    def serialize_prop(cls, max_len):
        data = bytearray()
        msg_pack_into('!HHH2x', data, 0, nicira_ext.NXAC2PT_MAX_LEN, 8, max_len)
        return data