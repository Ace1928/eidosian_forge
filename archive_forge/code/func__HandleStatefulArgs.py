from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute import zone_utils
from googlecloudsdk.api_lib.compute.instance_groups.managed import stateful_policy_utils as policy_utils
from googlecloudsdk.api_lib.compute.managed_instance_groups_utils import ValueOrNone
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import resource_manager_tags_utils
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as managed_flags
from googlecloudsdk.command_lib.compute.managed_instance_groups import auto_healing_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
import six
def _HandleStatefulArgs(self, instance_group_manager, args, client):
    instance_groups_flags.ValidateManagedInstanceGroupStatefulDisksProperties(args)
    instance_groups_flags.ValidateManagedInstanceGroupStatefulIPsProperties(args)
    if args.stateful_disk or args.stateful_internal_ip or args.stateful_external_ip:
        instance_group_manager.statefulPolicy = self._CreateStatefulPolicy(args, client)