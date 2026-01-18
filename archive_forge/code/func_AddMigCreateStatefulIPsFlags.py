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
def AddMigCreateStatefulIPsFlags(parser):
    """Adding stateful IPs flags to the parser."""
    stateful_internal_ips_help = textwrap.dedent('\n      Internal IPs considered stateful by the instance group. {}\n      Use this argument multiple times to make more internal IPs stateful.\n\n      At least one of the following is required:\n      {}\n      {}\n\n      Additional arguments:\n      {}\n      '.format(STATEFUL_IPS_HELP_BASE, STATEFUL_IP_ENABLED_ARG_HELP, STATEFUL_IP_INTERFACE_NAME_ARG_WITH_ENABLED_HELP, STATEFUL_IP_AUTO_DELETE_ARG_HELP))
    parser.add_argument('--stateful-internal-ip', type=arg_parsers.ArgDict(allow_key_only=True, spec={'enabled': None, 'interface-name': str, 'auto-delete': AutoDeleteFlag.ValidatorWithFlagName('--stateful-internal-ip')}), action='append', help=stateful_internal_ips_help)
    stateful_external_ips_help = textwrap.dedent('\n      External IPs considered stateful by the instance group. {}\n      Use this argument multiple times to make more external IPs stateful.\n\n      At least one of the following is required:\n      {}\n      {}\n\n      Additional arguments:\n      {}\n      '.format(STATEFUL_IPS_HELP_BASE, STATEFUL_IP_ENABLED_ARG_HELP, STATEFUL_IP_INTERFACE_NAME_ARG_WITH_ENABLED_HELP, STATEFUL_IP_AUTO_DELETE_ARG_HELP))
    parser.add_argument('--stateful-external-ip', type=arg_parsers.ArgDict(allow_key_only=True, spec={'enabled': None, 'interface-name': str, 'auto-delete': AutoDeleteFlag.ValidatorWithFlagName('--stateful-external-ip')}), action='append', help=stateful_external_ips_help)