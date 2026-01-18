from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.forwarding_rules import exceptions
from googlecloudsdk.command_lib.compute.forwarding_rules import flags
from googlecloudsdk.core import properties
def _ValidateRegionalTargetArgs(args):
    """Validate the regional forwarding rule target args.

  Args:
      args: The arguments given to the create/set-target command.
  """
    if getattr(args, 'global', None):
        raise exceptions.ArgumentError('You cannot specify [--global] for a regional forwarding rule.')
    if args.target_instance_zone and (not args.target_instance):
        raise exceptions.ArgumentError('You cannot specify [--target-instance-zone] unless you are specifying [--target-instance].')