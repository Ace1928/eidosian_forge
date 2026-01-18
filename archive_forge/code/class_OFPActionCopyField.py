import struct
import base64
from os_ken.lib import addrconv
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib.packet import packet
from os_ken import exception
from os_ken import utils
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase, MsgInMsgBase
from os_ken.ofproto import ether
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import nx_actions
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_v1_5 as ofproto
@OFPAction.register_action_type(ofproto.OFPAT_COPY_FIELD, ofproto.OFP_ACTION_COPY_FIELD_SIZE)
class OFPActionCopyField(OFPAction):
    """
    Copy Field action

    This action copy value between header and register.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    n_bits           Number of bits to copy.
    src_offset       Starting bit offset in source.
    dst_offset       Starting bit offset in destination.
    oxm_ids          List of ``OFPOxmId`` instances.
                     The first element of this list, src_oxm_id,
                     identifies the field where the value is copied from.
                     The second element of this list, dst_oxm_id,
                     identifies the field where the value is copied to.
                     The default is [].
    ================ ======================================================
    """

    def __init__(self, n_bits=0, src_offset=0, dst_offset=0, oxm_ids=None, type_=None, len_=None):
        oxm_ids = oxm_ids if oxm_ids else []
        super(OFPActionCopyField, self).__init__()
        self.n_bits = n_bits
        self.src_offset = src_offset
        self.dst_offset = dst_offset
        assert len(oxm_ids) == 2
        self.oxm_ids = []
        for i in oxm_ids:
            if isinstance(i, OFPOxmId):
                i.hasmask = False
                self.oxm_ids.append(i)
            elif isinstance(i, str):
                self.oxm_ids.append(OFPOxmId(i, hasmask=False))
            else:
                raise ValueError('invalid value for oxm_ids: %s' % oxm_ids)

    @classmethod
    def parser(cls, buf, offset):
        type_, len_, n_bits, src_offset, dst_offset = struct.unpack_from(ofproto.OFP_ACTION_COPY_FIELD_PACK_STR, buf, offset)
        offset += ofproto.OFP_ACTION_COPY_FIELD_SIZE
        rest = buf[offset:offset + len_]
        oxm_ids = []
        while rest:
            i, rest = OFPOxmId.parse(rest)
            oxm_ids.append(i)
        return cls(n_bits, src_offset, dst_offset, oxm_ids, type_, len_)

    def serialize(self, buf, offset):
        oxm_ids_buf = b''
        for i in self.oxm_ids:
            oxm_ids_buf += i.serialize()
        action_len = ofproto.OFP_ACTION_COPY_FIELD_SIZE + len(oxm_ids_buf)
        self.len = utils.round_up(action_len, 8)
        pad_len = self.len - action_len
        msg_pack_into(ofproto.OFP_ACTION_COPY_FIELD_PACK_STR, buf, offset, self.type, self.len, self.n_bits, self.src_offset, self.dst_offset)
        buf += oxm_ids_buf + b'\x00' * pad_len