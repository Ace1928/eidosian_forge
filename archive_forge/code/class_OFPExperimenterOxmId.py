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
class OFPExperimenterOxmId(OFPOxmId):

    def __init__(self, type_, exp_id, hasmask=False, length=None):
        super(OFPExperimenterOxmId, self).__init__(type_=type_, hasmask=hasmask, length=length)
        self.exp_id = exp_id

    def serialize(self):
        buf = super(OFPExperimenterOxmId, self).serialize()
        msg_pack_into(self._EXPERIMENTER_ID_PACK_STR, buf, struct.calcsize(self._PACK_STR), self.exp_id)