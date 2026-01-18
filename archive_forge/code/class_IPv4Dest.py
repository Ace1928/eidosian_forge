import logging
from os_ken.lib.packet.bgp import IPAddrPrefix
from os_ken.lib.packet.bgp import RF_IPv4_UC
from os_ken.services.protocols.bgp.info_base.base import Path
from os_ken.services.protocols.bgp.info_base.base import Table
from os_ken.services.protocols.bgp.info_base.base import Destination
from os_ken.services.protocols.bgp.info_base.base import NonVrfPathProcessingMixin
from os_ken.services.protocols.bgp.info_base.base import PrefixFilter
class IPv4Dest(Destination, NonVrfPathProcessingMixin):
    """VPNv4 Destination

    Store IPv4 Paths.
    """
    ROUTE_FAMILY = RF_IPv4_UC

    def _best_path_lost(self):
        old_best_path = self._best_path
        NonVrfPathProcessingMixin._best_path_lost(self)
        self._core_service._signal_bus.best_path_changed(old_best_path, True)

    def _new_best_path(self, best_path):
        NonVrfPathProcessingMixin._new_best_path(self, best_path)
        self._core_service._signal_bus.best_path_changed(best_path, False)