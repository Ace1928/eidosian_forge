from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.command_lib.compute import exceptions
from googlecloudsdk.core.console import console_io
def GetRegions(self, resource_refs):
    """Fetches region resources."""
    errors = []
    requests = []
    region_names = set()
    for resource_ref in resource_refs:
        if (resource_ref.project, resource_ref.region) not in region_names:
            region_names.add((resource_ref.project, resource_ref.region))
            requests.append((self.compute.regions, 'Get', self.messages.ComputeRegionsGetRequest(project=resource_ref.project, region=resource_ref.region)))
    if requests:
        res = list(request_helper.MakeRequests(requests=requests, http=self.http, batch_url=self.batch_url, errors=errors))
    else:
        return None
    if errors:
        return None
    else:
        return res