import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionController2PropUnknown(NXActionController2Prop):

    @classmethod
    def parser_prop(cls, buf, length):
        size = 4
        return (buf, size)

    @classmethod
    def serialize_prop(cls, argment):
        data = bytearray()
        return data