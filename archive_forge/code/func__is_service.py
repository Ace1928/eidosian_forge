import abc
import logging
from oslo_messaging.target import Target
def _is_service(self, address, service):
    return address.startswith(self._rpc_prefix if service == SERVICE_RPC else self._notify_prefix)