from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.gkeonprem import flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _AddVmwareHostConfig(vmware_network_config_group, for_update=False):
    """Adds flags to specify common parameters for all hosts irrespective of their IP address.

  Args:
    vmware_network_config_group: The parent group to add the flags to.
    for_update: bool, True to add flags for update command, False to add flags
      for create command.
  """
    if for_update:
        return
    vmware_host_config_group = vmware_network_config_group.add_group(help='Common parameters for all hosts irrespective of their IP address')
    vmware_host_config_group.add_argument('--dns-servers', metavar='DNS_SERVERS', type=arg_parsers.ArgList(str), help='DNS server IP address.')
    vmware_host_config_group.add_argument('--ntp-servers', metavar='NTP_SERVERS', type=arg_parsers.ArgList(str), help='NTP server IP address.')
    vmware_host_config_group.add_argument('--dns-search-domains', type=arg_parsers.ArgList(str), metavar='DNS_SEARCH_DOMAINS', help='DNS search domains.')