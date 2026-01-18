from os_ken.services.protocols.bgp.base import ActivityException
from os_ken.services.protocols.bgp.operator.command import Command
from os_ken.services.protocols.bgp.operator.command import CommandsResponse
from os_ken.services.protocols.bgp.operator.command import STATUS_ERROR
from os_ken.services.protocols.bgp.operator.command import STATUS_OK
from os_ken.services.protocols.bgp.operator.commands.responses import (
from .route_formatter_mixin import RouteFormatterMixin
class RibBase(Command, RouteFormatterMixin):
    supported_families = ['ipv4', 'ipv6', 'vpnv4', 'vpnv6', 'rtfilter', 'evpn', 'ipv4fs', 'ipv6fs', 'vpnv4fs', 'vpnv6fs', 'l2vpnfs']