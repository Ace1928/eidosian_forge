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
def AddMigUpdateStatefulFlagsIPs(parser):
    """Add stateful IPs flags to the parser."""
    stateful_ips_help_text_template = textwrap.dedent(STATEFUL_IPS_HELP_BASE + STATEFUL_IPS_HELP_TEMPLATE + '\n      At least one of the following is required:\n      {}\n      {}\n\n      Additional arguments:\n      {}\n      '.format(STATEFUL_IP_ENABLED_ARG_HELP, STATEFUL_IP_INTERFACE_NAME_ARG_WITH_ENABLED_HELP, STATEFUL_IP_AUTO_DELETE_ARG_HELP))
    stateful_internal_ip_flag_name = '--stateful-internal-ip'
    parser.add_argument(stateful_internal_ip_flag_name, type=arg_parsers.ArgDict(allow_key_only=True, spec={'enabled': None, 'interface-name': str, 'auto-delete': AutoDeleteFlag.ValidatorWithFlagName(stateful_internal_ip_flag_name)}), action='append', help=stateful_ips_help_text_template.format(ip_type='internal'))
    stateful_external_ip_flag_name = '--stateful-external-ip'
    parser.add_argument(stateful_external_ip_flag_name, type=arg_parsers.ArgDict(allow_key_only=True, spec={'enabled': None, 'interface-name': str, 'auto-delete': AutoDeleteFlag.ValidatorWithFlagName(stateful_external_ip_flag_name)}), action='append', help=stateful_ips_help_text_template.format(ip_type='external'))
    remove_stateful_ips_help_text_template = '\n      Remove stateful configuration for the specified interfaces for\n      {ip_type} IPs.\n      '
    parser.add_argument('--remove-stateful-internal-ips', metavar='INTERFACE_NAME', type=arg_parsers.ArgList(min_length=1), help=remove_stateful_ips_help_text_template.format(ip_type='internal'))
    parser.add_argument('--remove-stateful-external-ips', metavar='INTERFACE_NAME', type=arg_parsers.ArgList(min_length=1), help=remove_stateful_ips_help_text_template.format(ip_type='external'))