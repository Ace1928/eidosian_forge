import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionUnknown(NXAction):

    def __init__(self, subtype, data=None, type_=None, len_=None, experimenter=None):
        self._subtype = subtype
        super(NXActionUnknown, self).__init__()
        self.data = data

    @classmethod
    def parser(cls, buf):
        return cls(data=buf)

    def serialize_body(self):
        return bytearray() if self.data is None else self.data