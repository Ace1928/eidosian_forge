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
class EventAdjRibInChanged(EventBase):
    """
    Event called when any adj-RIB-in path is changed due to UPDATE messages
    or remote peer's down.

    This event is the wrapper for ``adj_rib_in_change_handler`` of
    ``bgpspeaker.BGPSpeaker``.

    ``path`` attribute contains an instance of ``info_base.base.Path``
    subclasses.

    If ``is_withdraw`` attribute is ``True``, ``path`` attribute has the
    information of the withdraw route.

    ``peer_ip`` is the peer's IP address who sent this path.

    ``peer_as`` is the peer's AS number who sent this path.
    """

    def __init__(self, path, is_withdraw, peer_ip, peer_as):
        super(EventAdjRibInChanged, self).__init__()
        self.path = path
        self.is_withdraw = is_withdraw
        self.peer_ip = peer_ip
        self.peer_as = peer_as