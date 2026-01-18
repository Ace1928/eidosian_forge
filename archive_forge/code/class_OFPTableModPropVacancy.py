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
@OFPTableModProp.register_type(ofproto.OFPTMPT_VACANCY)
class OFPTableModPropVacancy(OFPTableModProp):

    def __init__(self, type_=None, length=None, vacancy_down=None, vacancy_up=None, vacancy=None):
        self.type = type_
        self.length = length
        self.vacancy_down = vacancy_down
        self.vacancy_up = vacancy_up
        self.vacancy = vacancy

    @classmethod
    def parser(cls, buf):
        vacancy = cls()
        vacancy.type, vacancy.length, vacancy.vacancy_down, vacancy.vacancy_up, vacancy.vacancy = struct.unpack_from(ofproto.OFP_TABLE_MOD_PROP_VACANCY_PACK_STR, buf, 0)
        return vacancy

    def serialize(self):
        self.length = ofproto.OFP_TABLE_MOD_PROP_VACANCY_SIZE
        buf = bytearray()
        msg_pack_into(ofproto.OFP_TABLE_MOD_PROP_VACANCY_PACK_STR, buf, 0, self.type, self.length, self.vacancy_down, self.vacancy_up, self.vacancy)
        return buf