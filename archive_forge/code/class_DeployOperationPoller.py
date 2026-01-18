from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
class DeployOperationPoller(waiter.CloudOperationPoller):
    """Poller for Cloud Deploy operations API.

  This is necessary because the core operations library doesn't directly support
  simple_uri.
  """

    def __init__(self, client):
        """Initiates a DeployOperationPoller.

    Args:
      client: base_api.BaseApiClient, An instance of the Cloud Deploy client.
    """
        self.client = client
        super(DeployOperationPoller, self).__init__(self.client.client.projects_locations_operations, self.client.client.projects_locations_operations)

    def Poll(self, operation_ref):
        return self.client.Get(operation_ref)

    def GetResult(self, operation):
        return operation