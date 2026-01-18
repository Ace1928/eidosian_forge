import logging
from os_ken.lib.packet.bgp import IPAddrPrefix
from os_ken.lib.packet.bgp import RF_IPv4_UC
from os_ken.services.protocols.bgp.info_base.base import Path
from os_ken.services.protocols.bgp.info_base.base import Table
from os_ken.services.protocols.bgp.info_base.base import Destination
from os_ken.services.protocols.bgp.info_base.base import NonVrfPathProcessingMixin
from os_ken.services.protocols.bgp.info_base.base import PrefixFilter
class Ipv4PrefixFilter(PrefixFilter):
    """IPv4 Prefix Filter class"""
    ROUTE_FAMILY = RF_IPv4_UC