import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionSetTunnel(_NXActionSetTunnelBase):
    """
        Set Tunnel action

        This action sets the identifier (such as GRE) to the specified id.

        And equivalent to the followings action of ovs-ofctl command.

        .. note::
            This actions is supported by
            ``OFPActionSetField``
            in OpenFlow1.2 or later.

        ..
          set_tunnel:id
        ..

        +------------------------+
        | **set_tunnel**\\:\\ *id* |
        +------------------------+

        ================ ======================================================
        Attribute        Description
        ================ ======================================================
        tun_id           Tunnel ID(32bits)
        ================ ======================================================

        Example::

            actions += [parser.NXActionSetTunnel(tun_id=0xa)]
        """
    _subtype = nicira_ext.NXAST_SET_TUNNEL
    _fmt_str = '!2xI'