from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import resources
class VmwareClientBase(object):
    """Base class for vwmare API client wrappers."""

    def __init__(self, api_version=_DEFAULT_API_VERSION):
        self._client = apis.GetClientInstance('sddc', api_version)
        self._messages = apis.GetMessagesModule('sddc', api_version)
        self.service = None
        self.operations_service = self.client.projects_locations_operations

    @property
    def client(self):
        return self._client

    @property
    def messages(self):
        return self._messages

    def WaitForOperation(self, operation, message, is_delete=False):
        operation_ref = resources.REGISTRY.Parse(operation.name, collection='sddc.projects.locations.operations')
        if is_delete:
            poller = waiter.CloudOperationPollerNoResources(self.operations_service)
        else:
            poller = waiter.CloudOperationPoller(self.service, self.operations_service)
        return waiter.WaitFor(poller, operation_ref, message)