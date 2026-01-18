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
def _AddMigStatefulIPsFlags(parser, ip_argument_name, ip_help_text, remove_ip_argument_name, remove_ip_help_text):
    """Add args for per-instance configs update command."""
    parser.add_argument(ip_argument_name, type=arg_parsers.ArgDict(spec={'interface-name': str, 'address': str, 'auto-delete': AutoDeleteFlag.ValidatorWithFlagName(ip_argument_name)}), action='append', help=ip_help_text)
    parser.add_argument(remove_ip_argument_name, metavar='KEY', type=arg_parsers.ArgList(min_length=1), help=remove_ip_help_text)