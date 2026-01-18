from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def CheckOperationStatus(self, operation_dict, msg_template):
    """Checks operations status.

    Only logs the errors instead of re-throwing them.

    Args:
     operation_dict: dict[str, oOptional[clouddeploy_messages.Operation],
       dictionary of resource name and clouddeploy_messages.Operation. The
       operation can be None if the operation isn't executed.
     msg_template: output string template.
    """
    for resource_name, operation in operation_dict.items():
        if not operation or not operation.name:
            continue
        try:
            operation_ref = resources.REGISTRY.ParseRelativeName(operation.name, collection='clouddeploy.projects.locations.operations')
            _ = self.WaitForOperation(operation, operation_ref, 'Waiting for the operation on resource {}'.format(resource_name)).response
            log.status.Print(msg_template.format(resource_name))
        except core_exceptions.Error as e:
            log.status.Print('Operation failed: {}'.format(e))