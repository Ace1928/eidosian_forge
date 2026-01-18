from __future__ import annotations
import atexit
import time
import weakref
from typing import TYPE_CHECKING, Any, Mapping, Optional, cast
from pymongo import common, periodic_executor
from pymongo._csot import MovingMinimum
from pymongo.errors import NotPrimaryError, OperationFailure, _OperationCancelled
from pymongo.hello import Hello
from pymongo.lock import _create_lock
from pymongo.periodic_executor import _shutdown_executors
from pymongo.pool import _is_faas
from pymongo.read_preferences import MovingAverage
from pymongo.server_description import ServerDescription
from pymongo.srv_resolver import _SrvResolver
def _get_seedlist(self) -> Optional[list[tuple[str, Any]]]:
    """Poll SRV records for a seedlist.

        Returns a list of ServerDescriptions.
        """
    try:
        resolver = _SrvResolver(self._fqdn, self._settings.pool_options.connect_timeout, self._settings.srv_service_name)
        seedlist, ttl = resolver.get_hosts_and_min_ttl()
        if len(seedlist) == 0:
            raise Exception
    except Exception:
        self.request_check()
        return None
    else:
        self._executor.update_interval(max(ttl, common.MIN_SRV_RESCAN_INTERVAL))
        return seedlist