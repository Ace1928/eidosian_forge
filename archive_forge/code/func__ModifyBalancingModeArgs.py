from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.backend_services import backend_flags
from googlecloudsdk.command_lib.compute.backend_services import backend_services_utils
from googlecloudsdk.command_lib.compute.backend_services import flags
def _ModifyBalancingModeArgs(self, client, args, backend_to_update):
    """Update balancing mode fields in backend_to_update according to args.

    Args:
      client: The compute client.
      args: The arguments given to the update-backend command.
      backend_to_update: The backend message to modify.
    """
    _ModifyBalancingModeArgs(client.messages, args, backend_to_update)