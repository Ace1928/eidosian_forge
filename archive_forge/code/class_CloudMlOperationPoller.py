from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
class CloudMlOperationPoller(waiter.CloudOperationPoller):
    """Poller for Cloud ML Engine operations API.

  This is necessary because the core operations library doesn't directly support
  simple_uri.
  """

    def __init__(self, client):
        self.client = client
        super(CloudMlOperationPoller, self).__init__(self.client.client.projects_operations, self.client.client.projects_operations)

    def Poll(self, operation_ref):
        return self.client.Get(operation_ref)

    def GetResult(self, operation):
        return operation