import struct
import base64
from os_ken.lib import addrconv
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib.packet import packet
from os_ken import utils
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase, MsgInMsgBase
from os_ken.ofproto import ether
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import nx_actions
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_v1_4 as ofproto
class OFPTableDesc(StringifyMixin):

    def __init__(self, length=None, table_id=None, config=None, properties=None):
        super(OFPTableDesc, self).__init__()
        self.table_id = table_id
        self.length = length
        self.config = config
        self.properties = properties

    @classmethod
    def parser(cls, buf, offset):
        length, table_id, config = struct.unpack_from(ofproto.OFP_TABLE_DESC_PACK_STR, buf, offset)
        props = []
        rest = buf[offset + ofproto.OFP_TABLE_DESC_SIZE:offset + length]
        while rest:
            p, rest = OFPTableModProp.parse(rest)
            props.append(p)
        ofptabledesc = cls(length, table_id, config, props)
        return ofptabledesc