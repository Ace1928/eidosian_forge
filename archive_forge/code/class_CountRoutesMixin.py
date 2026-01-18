import logging
import pprint
from os_ken.services.protocols.bgp.operator.command import Command
from os_ken.services.protocols.bgp.operator.command import CommandsResponse
from os_ken.services.protocols.bgp.operator.command import STATUS_ERROR
from os_ken.services.protocols.bgp.operator.command import STATUS_OK
from os_ken.services.protocols.bgp.operator.commands.responses import \
from os_ken.services.protocols.bgp.operator.views.conf import ConfDetailView
from os_ken.services.protocols.bgp.operator.views.conf import ConfDictView
from .route_formatter_mixin import RouteFormatterMixin
class CountRoutesMixin(object):
    api = None

    def _count_routes(self, vrf_name, vrf_rf):
        return len(self.api.get_single_vrf_routes(vrf_name, vrf_rf))