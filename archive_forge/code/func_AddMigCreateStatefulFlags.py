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
def AddMigCreateStatefulFlags(parser):
    """Adding stateful flags for disks and names to the parser."""
    stateful_disks_help = textwrap.dedent(STATEFUL_DISKS_HELP_BASE + '\n      Use this argument multiple times to attach more disks.\n\n      *device-name*::: (Required) Device name of the disk to mark stateful.\n      ' + STATEFUL_DISK_AUTO_DELETE_ARG_HELP)
    parser.add_argument('--stateful-disk', type=arg_parsers.ArgDict(spec={'device-name': str, 'auto-delete': AutoDeleteFlag.ValidatorWithFlagName('--stateful_disk')}), action='append', help=stateful_disks_help)