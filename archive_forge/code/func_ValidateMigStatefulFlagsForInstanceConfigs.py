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
def ValidateMigStatefulFlagsForInstanceConfigs(args, for_update=False, need_disk_source=False):
    """Validates the values of stateful flags for instance configs."""
    ValidateMigStatefulDiskFlagForInstanceConfigs(args.stateful_disk, '--stateful-disk', for_update, need_disk_source)
    if for_update:
        ValidateMigStatefulDisksRemovalFlagForInstanceConfigs(disks_to_remove=args.remove_stateful_disks, disks_to_update=args.stateful_disk)
        ValidateMigStatefulMetadataRemovalFlagForInstanceConfigs(entries_to_remove=args.remove_stateful_metadata, entries_to_update=args.stateful_metadata)