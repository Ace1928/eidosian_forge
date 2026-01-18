import logging
from typing import Any, Dict, Optional
from ray import cloudpickle
from ray.serve._private.common import EndpointInfo, EndpointTag
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.long_poll import LongPollHost, LongPollNamespace
from ray.serve._private.storage.kv_store import KVStoreBase
def _notify_route_table_changed(self):
    self._long_poll_host.notify_changed(LongPollNamespace.ROUTE_TABLE, self._endpoints)