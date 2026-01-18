from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import functools
import ipaddress
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import containers_utils
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import disks_util
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import kms_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute.zones import service as zones_service
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as core_resources
import six
def AddInstanceTerminationActionVmArgs(parser, is_update=False):
    """Set arguments for specifing the termination action for the instance.

  For set_scheduling operation we are implementing this as argument group with
  additional argument clear-* providing the functionality to clear the
  instance-termination-action field.

  Args:
     parser: ArgumentParser, parser to which flags will be added.
     is_update: Bool. If True, flags are intended for set-scheduling operation.
  """
    if is_update:
        termination_action_group = parser.add_group('Instance Termination Action', mutex=True)
        termination_action_group.add_argument('--instance-termination-action', choices={'STOP': 'Default. Stop the VM without preserving memory. The VM can be restarted later.', 'DELETE': 'Permanently delete the VM.'}, type=arg_utils.ChoiceToEnumName, help='      Specifies the termination action that will be taken upon VM preemption\n      (`--provisioning-model=SPOT` or `--preemptible`) or automatic instance\n      termination (`--max-run-duration` or `--termination-time`).\n      ')
        termination_action_group.add_argument('--clear-instance-termination-action', action='store_true', help="        Disables the termination action for this VM if allowed OR\n        sets termination action to the default value.\n        Depending on a VM's availability settings, a termination action is\n        either required or not allowed. This flag is required when you are\n        updating a VM such that it's previously specified termination action is\n        no longer allowed.\n        If you use this flag when a VM requires a termination action,\n        it's termination action is just set to the default value (stop).\n        ")
    else:
        parser.add_argument('--instance-termination-action', choices={'STOP': 'Default. Stop the VM without preserving memory. The VM can be restarted later.', 'DELETE': 'Permanently delete the VM.'}, type=arg_utils.ChoiceToEnumName, help='      Specifies the termination action that will be taken upon VM preemption\n      (`--provisioning-model=SPOT` or `--preemptible`) or automatic instance\n      termination (`--max-run-duration` or `--termination-time`).\n      ')