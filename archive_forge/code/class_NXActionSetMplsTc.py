import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionSetMplsTc(NXAction):
    """
        Set MPLS Tc action

        This action sets the MPLS Tc.

        And equivalent to the followings action of ovs-ofctl command.

        ..
          set_mpls_tc:tc
        ..

        +-------------------------+
        | **set_mpls_tc**\\:\\ *tc* |
        +-------------------------+

        ================ ======================================================
        Attribute        Description
        ================ ======================================================
        tc               MPLS Tc
        ================ ======================================================

        .. NOTE::
            This actions is supported by
            ``OFPActionSetField(mpls_label=tc)``
            in OpenFlow1.2 or later.

        Example::

            actions += [parser.NXActionSetMplsLabel(tc=0x10)]
        """
    _subtype = nicira_ext.NXAST_SET_MPLS_TC
    _fmt_str = '!B5x'

    def __init__(self, tc, type_=None, len_=None, vendor=None, subtype=None):
        super(NXActionSetMplsTc, self).__init__()
        self.tc = tc

    @classmethod
    def parser(cls, buf):
        tc, = struct.unpack_from(cls._fmt_str, buf)
        return cls(tc)

    def serialize_body(self):
        data = bytearray()
        msg_pack_into(self._fmt_str, data, 0, self.tc)
        return data