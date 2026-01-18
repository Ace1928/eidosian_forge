import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
@NXActionController2Prop.register_type(nicira_ext.NXAC2PT_REASON)
class NXActionController2PropReason(NXActionController2Prop):
    _fmt_str = '!B3x'
    _arg_name = 'reason'

    @classmethod
    def parser_prop(cls, buf, length):
        size = 4
        reason, = struct.unpack_from(cls._fmt_str, buf, 0)
        return (reason, size)

    @classmethod
    def serialize_prop(cls, reason):
        data = bytearray()
        msg_pack_into('!HHB3x', data, 0, nicira_ext.NXAC2PT_REASON, 5, reason)
        return data