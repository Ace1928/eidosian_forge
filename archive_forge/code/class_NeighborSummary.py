import logging
from time import strftime
from os_ken.services.protocols.bgp.operator.command import Command
from os_ken.services.protocols.bgp.operator.command import CommandsResponse
from os_ken.services.protocols.bgp.operator.command import STATUS_ERROR
from os_ken.services.protocols.bgp.operator.command import STATUS_OK
from os_ken.services.protocols.bgp.operator.commands.responses import \
from os_ken.services.protocols.bgp.operator.views.bgp import CoreServiceDetailView
from os_ken.lib.packet.bgp import RF_IPv4_UC
from os_ken.lib.packet.bgp import RF_IPv6_UC
from os_ken.lib.packet.bgp import RF_IPv4_VPN
from os_ken.lib.packet.bgp import RF_IPv6_VPN
from os_ken.lib.packet.bgp import BGP_ATTR_ORIGIN_IGP
from os_ken.lib.packet.bgp import BGP_ATTR_ORIGIN_EGP
from os_ken.lib.packet.bgp import BGP_ATTR_ORIGIN_INCOMPLETE
class NeighborSummary(Command):
    help_msg = 'show summarized neighbor information'
    command = 'summary'

    def action(self, params):
        requested_peers = []
        if len(params) > 0:
            requested_peers = [str(p) for p in params]
        core_service = self.api.get_core_service()
        core_service_view = CoreServiceDetailView(core_service)
        peers_view = core_service_view.rel('peer_manager').rel('peers_summary')

        def filter_requested(peer_id, peer_obj):
            return not requested_peers or peer_id in requested_peers
        peers_view.apply_filter(filter_requested)
        ret = peers_view.encode()
        return CommandsResponse(STATUS_OK, ret)