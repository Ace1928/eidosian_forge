from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.ai import constants
class AiPlatformOperationPoller(waiter.CloudOperationPoller):
    """Poller for AI Platform operations API.

  This is necessary because the core operations library doesn't directly support
  simple_uri.
  """

    def __init__(self, client):
        self.client = client
        super(AiPlatformOperationPoller, self).__init__(self.client.client.projects_locations_operations, self.client.client.projects_locations_operations)

    def Poll(self, operation_ref):
        return self.client.Get(operation_ref)

    def GetResult(self, operation):
        return operation