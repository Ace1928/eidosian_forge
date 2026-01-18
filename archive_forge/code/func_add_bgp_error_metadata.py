import abc
from collections import OrderedDict
import logging
import socket
import time
import traceback
import weakref
import netaddr
from os_ken.lib import hub
from os_ken.lib import sockopt
from os_ken.lib import ip
from os_ken.lib.hub import Timeout
from os_ken.lib.packet.bgp import RF_IPv4_UC
from os_ken.lib.packet.bgp import RF_IPv6_UC
from os_ken.lib.packet.bgp import RF_IPv4_VPN
from os_ken.lib.packet.bgp import RF_IPv6_VPN
from os_ken.lib.packet.bgp import RF_L2_EVPN
from os_ken.lib.packet.bgp import RF_IPv4_FLOWSPEC
from os_ken.lib.packet.bgp import RF_IPv6_FLOWSPEC
from os_ken.lib.packet.bgp import RF_VPNv4_FLOWSPEC
from os_ken.lib.packet.bgp import RF_VPNv6_FLOWSPEC
from os_ken.lib.packet.bgp import RF_L2VPN_FLOWSPEC
from os_ken.lib.packet.bgp import RF_RTC_UC
from os_ken.services.protocols.bgp.utils.circlist import CircularListType
from os_ken.services.protocols.bgp.utils.evtlet import LoopingCall
def add_bgp_error_metadata(code, sub_code, def_desc='unknown'):
    """Decorator for all exceptions that want to set exception class meta-data.
    """
    if _EXCEPTION_REGISTRY.get((code, sub_code)) is not None:
        raise ValueError('BGPSException with code %d and sub-code %d already defined.' % (code, sub_code))

    def decorator(subclass):
        """Sets class constants for exception code and sub-code.

        If given class is sub-class of BGPSException we sets class constants.
        """
        if issubclass(subclass, BGPSException):
            _EXCEPTION_REGISTRY[code, sub_code] = subclass
            subclass.CODE = code
            subclass.SUB_CODE = sub_code
            subclass.DEF_DESC = def_desc
        return subclass
    return decorator