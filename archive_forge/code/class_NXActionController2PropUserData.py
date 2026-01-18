import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
@NXActionController2Prop.register_type(nicira_ext.NXAC2PT_USERDATA)
class NXActionController2PropUserData(NXActionController2Prop):
    _fmt_str = '!B'
    _arg_name = 'userdata'

    @classmethod
    def parser_prop(cls, buf, length):
        userdata = []
        offset = 0
        while offset < length:
            u = struct.unpack_from(cls._fmt_str, buf, offset)
            userdata.append(u[0])
            offset += 1
        user_size = utils.round_up(length, 4)
        if user_size > 4 and user_size % 8 == 0:
            size = utils.round_up(length, 4) + 4
        else:
            size = utils.round_up(length, 4)
        return (userdata, size)

    @classmethod
    def serialize_prop(cls, userdata):
        data = bytearray()
        user_buf = bytearray()
        user_offset = 0
        for user in userdata:
            msg_pack_into('!B', user_buf, user_offset, user)
            user_offset += 1
        msg_pack_into('!HH', data, 0, nicira_ext.NXAC2PT_USERDATA, 4 + user_offset)
        data += user_buf
        if user_offset > 4:
            user_len = utils.round_up(user_offset, 4)
            brank_size = 0
            if user_len % 8 == 0:
                brank_size = 4
            msg_pack_into('!%dx' % (user_len - user_offset + brank_size), data, 4 + user_offset)
        else:
            user_len = utils.round_up(user_offset, 4)
            msg_pack_into('!%dx' % (user_len - user_offset), data, 4 + user_offset)
        return data