from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
def WaitOperation(self, operation, operation_poller=None, message=None):
    if not operation_poller:
        operation_poller = poller.Poller(self._service, self.ref, has_project=self._api_has_project)
    if self._op_has_project and 'projects' not in operation.selfLink:
        operation.selfLink = operation.selfLink.replace('locations', 'projects/locations')
    operation_ref = self._resources.Parse(operation.selfLink, collection=OP_COLLECTION_NAME)
    return waiter.WaitFor(operation_poller, operation_ref, message)