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
def _AddInstanceGroupManagerArgs(parser):
    """Adds args."""
    parser.add_argument('--base-instance-name', help='Base name to use for the Compute Engine instances that will be created with the managed instance group. If not provided base instance name will be the prefix of instance group name.')
    parser.add_argument('--size', required=True, type=arg_parsers.BoundedInt(0, sys.maxsize, unlimited=True), help='Initial number of instances you want in this group.')
    instance_groups_flags.AddDescriptionFlag(parser)
    parser.add_argument('--target-pool', type=arg_parsers.ArgList(), metavar='TARGET_POOL', help='Specifies any target pools you want the instances of this managed instance group to be part of.')
    managed_flags.INSTANCE_TEMPLATE_ARG.AddArgument(parser)