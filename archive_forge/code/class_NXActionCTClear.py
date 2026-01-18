import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionCTClear(NXAction):
    """
        Clear connection tracking state action

        This action clears connection tracking state from packets.

        And equivalent to the followings action of ovs-ofctl command.

        ..
          ct_clear
        ..

        +--------------+
        | **ct_clear** |
        +--------------+

        Example::

            actions += [parser.NXActionCTClear()]
        """
    _subtype = nicira_ext.NXAST_CT_CLEAR
    _fmt_str = '!6x'

    def __init__(self, type_=None, len_=None, experimenter=None, subtype=None):
        super(NXActionCTClear, self).__init__()

    @classmethod
    def parser(cls, buf):
        return cls()

    def serialize_body(self):
        data = bytearray()
        msg_pack_into(self._fmt_str, data, 0)
        return data