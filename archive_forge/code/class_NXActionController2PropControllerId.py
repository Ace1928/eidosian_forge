import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
@NXActionController2Prop.register_type(nicira_ext.NXAC2PT_CONTROLLER_ID)
class NXActionController2PropControllerId(NXActionController2Prop):
    _fmt_str = '!H2x'
    _arg_name = 'controller_id'

    @classmethod
    def parser_prop(cls, buf, length):
        size = 4
        controller_id, = struct.unpack_from(cls._fmt_str, buf, 0)
        return (controller_id, size)

    @classmethod
    def serialize_prop(cls, controller_id):
        data = bytearray()
        msg_pack_into('!HHH2x', data, 0, nicira_ext.NXAC2PT_CONTROLLER_ID, 8, controller_id)
        return data