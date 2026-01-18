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
def _validate_callable(self, callable_):
    if callable_ is None:
        raise ActivityException(desc='Callable cannot be None')
    if not hasattr(callable_, '__call__'):
        raise ActivityException(desc='Currently only supports instances that have __call__ as callable which is missing in given arg.')
    if not self._started:
        raise ActivityException(desc='Tried to spawn a child thread before this Activity was started.')