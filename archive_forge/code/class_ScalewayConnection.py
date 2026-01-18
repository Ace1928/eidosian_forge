import copy
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError
from libcloud.compute.base import (
from libcloud.compute.types import NodeState, VolumeSnapshotState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
class ScalewayConnection(ConnectionUserAndKey):
    """
    Connection class for the Scaleway driver.
    """
    host = SCALEWAY_API_HOSTS['default']
    allow_insecure = False
    responseCls = ScalewayResponse

    def request(self, action, params=None, data=None, headers=None, method='GET', raw=False, stream=False, region=None):
        if region:
            old_host = self.host
            self.host = SCALEWAY_API_HOSTS[region.id if isinstance(region, NodeLocation) else region]
            if not self.host == old_host:
                self.connect()
        return super().request(action, params, data, headers, method, raw, stream)

    def _request_paged(self, action, params=None, data=None, headers=None, method='GET', raw=False, stream=False, region=None):
        if params is None:
            params = {}
        if isinstance(params, dict):
            params['per_page'] = 100
        elif isinstance(params, list):
            params.append(('per_page', 100))
        results = self.request(action, params, data, headers, method, raw, stream, region).object
        links = self.connection.getresponse().links
        while links and 'next' in links:
            next = self.request(links['next']['url'], data=data, headers=headers, method=method, raw=raw, stream=stream).object
            links = self.connection.getresponse().links
            merged = {root: child + next[root] for root, child in list(results.items())}
            results = merged
        return results

    def add_default_headers(self, headers):
        """
        Add headers that are necessary for every request
        """
        headers['X-Auth-Token'] = self.key
        headers['Content-Type'] = 'application/json'
        return headers