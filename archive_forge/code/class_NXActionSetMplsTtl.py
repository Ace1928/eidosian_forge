import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionSetMplsTtl(NXAction):
    """
        Set MPLS TTL action

        This action sets the MPLS TTL.

        And equivalent to the followings action of ovs-ofctl command.

        ..
          set_mpls_ttl:ttl
        ..

        +---------------------------+
        | **set_mpls_ttl**\\:\\ *ttl* |
        +---------------------------+

        ================ ======================================================
        Attribute        Description
        ================ ======================================================
        ttl              MPLS TTL
        ================ ======================================================

        .. NOTE::
            This actions is supported by
            ``OFPActionSetMplsTtl``
            in OpenFlow1.2 or later.

        Example::

            actions += [parser.NXActionSetMplsTil(ttl=128)]
        """
    _subtype = nicira_ext.NXAST_SET_MPLS_TTL
    _fmt_str = '!B5x'

    def __init__(self, ttl, type_=None, len_=None, vendor=None, subtype=None):
        super(NXActionSetMplsTtl, self).__init__()
        self.ttl = ttl

    @classmethod
    def parser(cls, buf):
        ttl, = struct.unpack_from(cls._fmt_str, buf)
        return cls(ttl)

    def serialize_body(self):
        data = bytearray()
        msg_pack_into(self._fmt_str, data, 0, self.ttl)
        return data