import abc
import logging
from oslo_messaging.target import Target
def _is_anycast(self, address):
    return address.startswith(self._rpc_anycast) or address.startswith(self._notify_anycast)