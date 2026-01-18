import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionRegMove(NXAction):
    """
        Move register action

        This action copies the src to dst.

        And equivalent to the followings action of ovs-ofctl command.

        ..
          move:src[start..end]->dst[start..end]
        ..

        +--------------------------------------------------------+
        | **move**\\:\\ *src*\\ **[**\\ *start*\\..\\ *end*\\ **]**\\->\\ |
        | *dst*\\ **[**\\ *start*\\..\\ *end* \\ **]**                |
        +--------------------------------------------------------+

        ================ ======================================================
        Attribute        Description
        ================ ======================================================
        src_field        OXM/NXM header for source field
        dst_field        OXM/NXM header for destination field
        n_bits           Number of bits
        src_ofs          Starting bit offset in source
        dst_ofs          Starting bit offset in destination
        ================ ======================================================

        .. CAUTION::
            **src_start**\\  and \\ **src_end**\\  difference and \\ **dst_start**\\
             and \\ **dst_end**\\  difference must be the same.

        Example::

            actions += [parser.NXActionRegMove(src_field="reg0",
                                               dst_field="reg1",
                                               n_bits=5,
                                               src_ofs=0
                                               dst_ofs=10)]
        """
    _subtype = nicira_ext.NXAST_REG_MOVE
    _fmt_str = '!HHH'
    _TYPE = {'ascii': ['src_field', 'dst_field']}

    def __init__(self, src_field, dst_field, n_bits, src_ofs=0, dst_ofs=0, type_=None, len_=None, experimenter=None, subtype=None):
        super(NXActionRegMove, self).__init__()
        self.n_bits = n_bits
        self.src_ofs = src_ofs
        self.dst_ofs = dst_ofs
        self.src_field = src_field
        self.dst_field = dst_field

    @classmethod
    def parser(cls, buf):
        n_bits, src_ofs, dst_ofs = struct.unpack_from(cls._fmt_str, buf, 0)
        rest = buf[struct.calcsize(NXActionRegMove._fmt_str):]
        n, len = ofp.oxm_parse_header(rest, 0)
        src_field = ofp.oxm_to_user_header(n)
        rest = rest[len:]
        n, len = ofp.oxm_parse_header(rest, 0)
        dst_field = ofp.oxm_to_user_header(n)
        rest = rest[len:]
        return cls(src_field, dst_field=dst_field, n_bits=n_bits, src_ofs=src_ofs, dst_ofs=dst_ofs)

    def serialize_body(self):
        data = bytearray()
        msg_pack_into(self._fmt_str, data, 0, self.n_bits, self.src_ofs, self.dst_ofs)
        n = ofp.oxm_from_user_header(self.src_field)
        ofp.oxm_serialize_header(n, data, len(data))
        n = ofp.oxm_from_user_header(self.dst_field)
        ofp.oxm_serialize_header(n, data, len(data))
        return data