import logging
import socket
import traceback
import msgpack
from os_ken.lib.packet import safi as subaddr_family
from os_ken.services.protocols.bgp import api
from os_ken.services.protocols.bgp.api.base import ApiException
from os_ken.services.protocols.bgp.api.base import NEXT_HOP
from os_ken.services.protocols.bgp.api.base import ORIGIN_RD
from os_ken.services.protocols.bgp.api.base import PREFIX
from os_ken.services.protocols.bgp.api.base import ROUTE_DISTINGUISHER
from os_ken.services.protocols.bgp.api.base import VPN_LABEL
from os_ken.services.protocols.bgp.base import Activity
from os_ken.services.protocols.bgp.base import add_bgp_error_metadata
from os_ken.services.protocols.bgp.base import BGPSException
from os_ken.services.protocols.bgp.base import FlexinetPeer
from os_ken.services.protocols.bgp.base import NET_CTRL_ERROR_CODE
from os_ken.services.protocols.bgp.constants import VRF_TABLE
from os_ken.services.protocols.bgp.rtconf.vrfs import VRF_RF
from os_ken.services.protocols.bgp.rtconf.vrfs import VrfConf
from os_ken.services.protocols.bgp.utils.validation import is_valid_ipv4
def _process_outgoing_msg(self, sink_iter):
    """For every message we construct a corresponding RPC message to be
        sent over the given socket inside given RPC session.

        This function should be launched in a new green thread as
        it loops forever.
        """
    LOG.debug('NetworkController processing outgoing request list.')
    from os_ken.services.protocols.bgp.model import FlexinetOutgoingRoute
    while self.is_connected:
        for outgoing_msg in sink_iter:
            if not self.is_connected:
                self._socket.close()
                return
            if isinstance(outgoing_msg, FlexinetOutgoingRoute):
                rpc_msg = _create_prefix_notification(outgoing_msg, self)
            else:
                raise NotImplementedError('Do not handle out going message of type %s' % outgoing_msg.__class__)
            if rpc_msg:
                self._sendall(rpc_msg)
        self.pause(0)
    if self.green_in:
        self.green_in.kill()