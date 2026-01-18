from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import textwrap
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as mig_flags
from googlecloudsdk.command_lib.util import completers
import six
def ValidateMigStatefulDisksRemovalFlagForInstanceConfigs(disks_to_remove, disks_to_update):
    remove_stateful_disks_set = set(disks_to_remove or [])
    for stateful_disk_to_update in disks_to_update or []:
        if stateful_disk_to_update.get('device-name') in remove_stateful_disks_set:
            raise exceptions.InvalidArgumentException(parameter_name='--remove-stateful-disks', message='the same [device-name] `{0}` cannot be updated and removed in one command call'.format(stateful_disk_to_update.get('device-name')))