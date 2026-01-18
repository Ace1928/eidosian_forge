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
def _process_incoming_msgs(self):
    LOG.debug('NetworkController started processing incoming messages')
    assert self._socket
    while self.is_connected:
        msg_buff = self._recv()
        if len(msg_buff) == 0:
            LOG.info('Peer %s disconnected.', self.peer_name)
            self.is_connected = False
            self._socket.close()
            break
        messages = self.feed_and_get_messages(msg_buff)
        for msg in messages:
            if msg[0] == RPC_MSG_REQUEST:
                try:
                    result = _handle_request(msg)
                    self._send_success_response(msg, result)
                except BGPSException as e:
                    self._send_error_response(msg, e.message)
            elif msg[0] == RPC_MSG_RESPONSE:
                _handle_response(msg)
            elif msg[0] == RPC_MSG_NOTIFY:
                _handle_notification(msg)
            else:
                LOG.error('Invalid message type: %r', msg)
            self.pause(0)
    if self.green_out:
        self.green_out.kill()