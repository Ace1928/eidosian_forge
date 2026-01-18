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
def AddMigStatefulFlagsForUpdateInstanceConfigs(parser):
    """Add args for per-instance configs update command."""
    _AddMigStatefulInstanceConfigsInstanceArg(parser)
    stateful_disk_argument_name = '--stateful-disk'
    disk_help_text = textwrap.dedent(STATEFUL_DISKS_HELP_INSTANCE_CONFIGS_UPDATE + STATEFUL_DISK_DEVICE_NAME_ARG_HELP + STATEFUL_DISK_SOURCE_ARG_HELP + STATEFUL_DISK_MODE_ARG_HELP + STATEFUL_DISK_AUTO_DELETE_ARG_HELP)
    parser.add_argument(stateful_disk_argument_name, type=arg_parsers.ArgDict(spec={'device-name': str, 'source': str, 'mode': str, 'auto-delete': AutoDeleteFlag.ValidatorWithFlagName(stateful_disk_argument_name)}), action='append', help=disk_help_text)
    parser.add_argument('--remove-stateful-disks', metavar='DEVICE_NAME', type=arg_parsers.ArgList(min_length=1), help="Remove stateful configuration for the specified disks from the instance's configuration.")
    stateful_metadata_argument_name = '--stateful-metadata'
    metadata_help_text = textwrap.dedent((STATEFUL_METADATA_HELP + STATEFUL_METADATA_HELP_UPDATE).format(argument_name=stateful_metadata_argument_name))
    parser.add_argument(stateful_metadata_argument_name, type=arg_parsers.ArgDict(min_length=1), default={}, action=arg_parsers.StoreOnceAction, metavar='KEY=VALUE', help=textwrap.dedent(metadata_help_text))
    parser.add_argument('--remove-stateful-metadata', metavar='KEY', type=arg_parsers.ArgList(min_length=1), help="Remove stateful configuration for the specified metadata keys from the instance's configuration.")