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
class SrvMonitor(MonitorBase):

    def __init__(self, topology: Topology, topology_settings: TopologySettings):
        """Class to poll SRV records on a background thread.

        Pass a Topology and a TopologySettings.

        The Topology is weakly referenced.
        """
        super().__init__(topology, 'pymongo_srv_polling_thread', common.MIN_SRV_RESCAN_INTERVAL, topology_settings.heartbeat_frequency)
        self._settings = topology_settings
        self._seedlist = self._settings._seeds
        assert isinstance(self._settings.fqdn, str)
        self._fqdn: str = self._settings.fqdn

    def _run(self) -> None:
        seedlist = self._get_seedlist()
        if seedlist:
            self._seedlist = seedlist
            try:
                self._topology.on_srv_update(self._seedlist)
            except ReferenceError:
                self.close()

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