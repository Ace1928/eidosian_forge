from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as instance_groups_managed_flags
from googlecloudsdk.command_lib.compute.instance_groups.managed import rolling_action
def _AddArgs(parser, supports_min_ready=False):
    """Adds args."""
    instance_groups_managed_flags.AddMaxUnavailableArg(parser)
    if supports_min_ready:
        instance_groups_managed_flags.AddMinReadyArg(parser)