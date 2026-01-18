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
def AddMigStatefulIPsFlagsForInstanceConfigs(parser):
    """Adding stateful IPs flags for creating instance configs."""
    stateful_ip_help = textwrap.dedent('\n      {}\n      Use this argument multiple times to attach and preserve multiple IPs.\n\n      {}\n      {}\n      {}\n      '.format(STATEFUL_IPS_HELP_INSTANCE_CONFIGS, STATEFUL_IP_INTERFACE_NAME_ARG_WITH_ADDRESS_HELP, STATEFUL_IP_ADDRESS_ARG_HELP, STATEFUL_IP_AUTO_DELETE_ARG_HELP))
    stateful_internal_ip_argument_name = '--stateful-internal-ip'
    parser.add_argument(stateful_internal_ip_argument_name, type=arg_parsers.ArgDict(spec={'interface-name': str, 'address': str, 'auto-delete': AutoDeleteFlag.ValidatorWithFlagName(stateful_internal_ip_argument_name)}), action='append', help=stateful_ip_help)
    stateful_external_ip_argument_name = '--stateful-external-ip'
    parser.add_argument(stateful_external_ip_argument_name, type=arg_parsers.ArgDict(spec={'interface-name': str, 'address': str, 'auto-delete': AutoDeleteFlag.ValidatorWithFlagName(stateful_external_ip_argument_name)}), action='append', help=stateful_ip_help)