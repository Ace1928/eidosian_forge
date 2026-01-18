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
def _sock_wrap(self, func):

    def wrapper(*args, **kwargs):
        try:
            ret = func(*args, **kwargs)
        except socket.error:
            LOG.error(traceback.format_exc())
            self._socket_error()
            return
        return ret
    return wrapper