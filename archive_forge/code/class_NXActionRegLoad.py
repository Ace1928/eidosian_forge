import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionRegLoad(NXAction):
    """
        Load literal value action

        This action loads a literal value into a field or part of a field.

        And equivalent to the followings action of ovs-ofctl command.

        ..
          load:value->dst[start..end]
        ..

        +-----------------------------------------------------------------+
        | **load**\\:\\ *value*\\->\\ *dst*\\ **[**\\ *start*\\..\\ *end*\\ **]**  |
        +-----------------------------------------------------------------+

        ================ ======================================================
        Attribute        Description
        ================ ======================================================
        ofs_nbits        Start and End for the OXM/NXM field.
                         Setting method refer to the ``nicira_ext.ofs_nbits``
        dst              OXM/NXM header for destination field
        value            OXM/NXM value to be loaded
        ================ ======================================================

        Example::

            actions += [parser.NXActionRegLoad(
                            ofs_nbits=nicira_ext.ofs_nbits(4, 31),
                            dst="eth_dst",
                            value=0x112233)]
        """
    _subtype = nicira_ext.NXAST_REG_LOAD
    _fmt_str = '!HIQ'
    _TYPE = {'ascii': ['dst']}

    def __init__(self, ofs_nbits, dst, value, type_=None, len_=None, experimenter=None, subtype=None):
        super(NXActionRegLoad, self).__init__()
        self.ofs_nbits = ofs_nbits
        self.dst = dst
        self.value = value

    @classmethod
    def parser(cls, buf):
        ofs_nbits, dst, value = struct.unpack_from(cls._fmt_str, buf, 0)
        dst_name = ofp.oxm_to_user_header(dst >> 9)
        return cls(ofs_nbits, dst_name, value)

    def serialize_body(self):
        hdr_data = bytearray()
        n = ofp.oxm_from_user_header(self.dst)
        ofp.oxm_serialize_header(n, hdr_data, 0)
        dst_num, = struct.unpack_from('!I', bytes(hdr_data), 0)
        data = bytearray()
        msg_pack_into(self._fmt_str, data, 0, self.ofs_nbits, dst_num, self.value)
        return data