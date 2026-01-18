from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instances import exceptions
from googlecloudsdk.command_lib.compute.instances import flags
def _scopes(self, args, instance_ref, client):
    """Get list of scopes to be assigned to the instance.

    Args:
      args: parsed command  line arguments.
      instance_ref: reference to the instance to which scopes will be assigned.
      client: a compute_holder.client instance

    Returns:
      List of scope urls extracted from args, with scope aliases expanded.
    """
    result = []
    for unprocessed_scope in self._unprocessed_scopes(args, instance_ref, client):
        scope = constants.SCOPES.get(unprocessed_scope, [unprocessed_scope])
        result.extend(scope)
    return result