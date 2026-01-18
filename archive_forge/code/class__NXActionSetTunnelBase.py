import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class _NXActionSetTunnelBase(NXAction):

    def __init__(self, tun_id, type_=None, len_=None, experimenter=None, subtype=None):
        super(_NXActionSetTunnelBase, self).__init__()
        self.tun_id = tun_id

    @classmethod
    def parser(cls, buf):
        tun_id, = struct.unpack_from(cls._fmt_str, buf, 0)
        return cls(tun_id)

    def serialize_body(self):
        data = bytearray()
        msg_pack_into(self._fmt_str, data, 0, self.tun_id)
        return data