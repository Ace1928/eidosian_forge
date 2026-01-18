import abc
import logging
from oslo_messaging.target import Target
def _is_unicast(self, address):
    return address.startswith(self._rpc_unicast) or address.startswith(self._notify_unicast)