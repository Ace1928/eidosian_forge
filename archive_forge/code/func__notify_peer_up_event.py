import logging
import os
from os_ken import cfg
from os_ken.lib import hub
from os_ken.utils import load_source
from os_ken.base.app_manager import OSKenApp
from os_ken.controller.event import EventBase
from os_ken.services.protocols.bgp.base import add_bgp_error_metadata
from os_ken.services.protocols.bgp.base import BGPSException
from os_ken.services.protocols.bgp.base import BIN_ERROR
from os_ken.services.protocols.bgp.bgpspeaker import BGPSpeaker
from os_ken.services.protocols.bgp.net_ctrl import NET_CONTROLLER
from os_ken.services.protocols.bgp.net_ctrl import NC_RPC_BIND_IP
from os_ken.services.protocols.bgp.net_ctrl import NC_RPC_BIND_PORT
from os_ken.services.protocols.bgp.rtconf.base import RuntimeConfigError
from os_ken.services.protocols.bgp.rtconf.common import LOCAL_AS
from os_ken.services.protocols.bgp.rtconf.common import ROUTER_ID
from os_ken.services.protocols.bgp.utils.validation import is_valid_ipv4
from os_ken.services.protocols.bgp.utils.validation import is_valid_ipv6
def _notify_peer_up_event(self, remote_ip, remote_as):
    ev = EventPeerUp(remote_ip, remote_as)
    self.send_event_to_observers(ev)