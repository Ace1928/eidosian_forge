from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.iam import iam_util
def _MakeGetRequestTuple(self):
    region = getattr(self.ref, 'region', None)
    if region is not None:
        return (self._client.regionBackendServices, 'Get', self._messages.ComputeRegionBackendServicesGetRequest(project=self.ref.project, region=region, backendService=self.ref.Name()))
    else:
        return (self._client.backendServices, 'Get', self._messages.ComputeBackendServicesGetRequest(project=self.ref.project, backendService=self.ref.Name()))