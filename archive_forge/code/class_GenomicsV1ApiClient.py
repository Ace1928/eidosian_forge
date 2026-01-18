from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import re
from googlecloudsdk.api_lib.genomics import genomics_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import resources
import six
class GenomicsV1ApiClient(GenomicsApiClient):
    """Client for accessing the V1 genomics API.
  """

    def __init__(self):
        super(GenomicsV1ApiClient, self).__init__('v1')

    def ResourceFromName(self, name):
        return self._registry.Parse(name, collection='genomics.operations')

    def Poller(self):
        return waiter.CloudOperationPollerNoResources(self._client.operations)

    def GetOperation(self, resource):
        return self._client.operations.Get(self._messages.GenomicsOperationsGetRequest(name=resource.RelativeName()))

    def CancelOperation(self, resource):
        return self._client.operations.Cancel(self._messages.GenomicsOperationsCancelRequest(name=resource.RelativeName()))