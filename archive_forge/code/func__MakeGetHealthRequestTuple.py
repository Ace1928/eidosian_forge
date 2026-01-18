from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.iam import iam_util
def _MakeGetHealthRequestTuple(self, group):
    region = getattr(self.ref, 'region', None)
    if region is not None:
        return (self._client.regionBackendServices, 'GetHealth', self._messages.ComputeRegionBackendServicesGetHealthRequest(resourceGroupReference=self._messages.ResourceGroupReference(group=group), project=self.ref.project, region=region, backendService=self.ref.Name()))
    else:
        return (self._client.backendServices, 'GetHealth', self._messages.ComputeBackendServicesGetHealthRequest(resourceGroupReference=self._messages.ResourceGroupReference(group=group), project=self.ref.project, backendService=self.ref.Name()))