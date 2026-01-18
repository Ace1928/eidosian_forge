import threading
from openstack import exceptions
def _find_interesting_networks(self):
    if self._networks_lock.acquire():
        try:
            if self._network_list_stamp:
                return
            if not self._use_external_network and (not self._use_internal_network):
                return
            if not self.has_service('network'):
                return
            self._set_interesting_networks()
            self._network_list_stamp = True
        finally:
            self._networks_lock.release()