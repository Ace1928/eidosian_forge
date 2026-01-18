from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.command_lib.compute import exceptions
from googlecloudsdk.core.console import console_io
def GetZones(self, resource_refs):
    """Fetches zone resources."""
    errors = []
    requests = []
    zone_names = set()
    for resource_ref in resource_refs:
        if resource_ref.zone not in zone_names:
            zone_names.add(resource_ref.zone)
            requests.append((self._compute.zones, 'Get', self._messages.ComputeZonesGetRequest(project=resource_ref.project, zone=resource_ref.zone)))
    res = list(request_helper.MakeRequests(requests=requests, http=self._http, batch_url=self._batch_url, errors=errors))
    if errors:
        return None
    else:
        return res