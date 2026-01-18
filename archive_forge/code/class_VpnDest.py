import abc
import logging
from os_ken.lib.packet.bgp import RF_L2_EVPN
from os_ken.services.protocols.bgp.info_base.base import Destination
from os_ken.services.protocols.bgp.info_base.base import NonVrfPathProcessingMixin
from os_ken.services.protocols.bgp.info_base.base import Path
from os_ken.services.protocols.bgp.info_base.base import Table
class VpnDest(Destination, NonVrfPathProcessingMixin, metaclass=abc.ABCMeta):
    """Base class for VPN destinations."""

    def _best_path_lost(self):
        old_best_path = self._best_path
        NonVrfPathProcessingMixin._best_path_lost(self)
        self._core_service._signal_bus.best_path_changed(old_best_path, True)
        if old_best_path:
            withdraw_clone = old_best_path.clone(for_withdrawal=True)
            tm = self._core_service.table_manager
            tm.import_single_vpn_path_to_all_vrfs(withdraw_clone, path_rts=old_best_path.get_rts())

    def _new_best_path(self, best_path):
        NonVrfPathProcessingMixin._new_best_path(self, best_path)
        self._core_service._signal_bus.best_path_changed(best_path, False)
        tm = self._core_service.table_manager
        tm.import_single_vpn_path_to_all_vrfs(self._best_path, self._best_path.get_rts())