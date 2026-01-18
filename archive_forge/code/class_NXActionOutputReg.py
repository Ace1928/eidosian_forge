import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionOutputReg(NXAction):
    """
        Add output action

        This action outputs the packet to the OpenFlow port number read from
        src.

        And equivalent to the followings action of ovs-ofctl command.

        ..
          output:src[start...end]
        ..

        +-------------------------------------------------------+
        | **output**\\:\\ *src*\\ **[**\\ *start*\\...\\ *end*\\ **]** |
        +-------------------------------------------------------+

        ================ ======================================================
        Attribute        Description
        ================ ======================================================
        ofs_nbits        Start and End for the OXM/NXM field.
                         Setting method refer to the ``nicira_ext.ofs_nbits``
        src              OXM/NXM header for source field
        max_len          Max length to send to controller
        ================ ======================================================

        Example::

            actions += [parser.NXActionOutputReg(
                            ofs_nbits=nicira_ext.ofs_nbits(4, 31),
                            src="reg0",
                            max_len=1024)]
        """
    _subtype = nicira_ext.NXAST_OUTPUT_REG
    _fmt_str = '!H4sH6x'
    _TYPE = {'ascii': ['src']}

    def __init__(self, ofs_nbits, src, max_len, type_=None, len_=None, experimenter=None, subtype=None):
        super(NXActionOutputReg, self).__init__()
        self.ofs_nbits = ofs_nbits
        self.src = src
        self.max_len = max_len

    @classmethod
    def parser(cls, buf):
        ofs_nbits, oxm_data, max_len = struct.unpack_from(cls._fmt_str, buf, 0)
        n, len_ = ofp.oxm_parse_header(oxm_data, 0)
        src = ofp.oxm_to_user_header(n)
        return cls(ofs_nbits, src, max_len)

    def serialize_body(self):
        data = bytearray()
        src = bytearray()
        oxm = ofp.oxm_from_user_header(self.src)
        (ofp.oxm_serialize_header(oxm, src, 0),)
        msg_pack_into(self._fmt_str, data, 0, self.ofs_nbits, bytes(src), self.max_len)
        return data