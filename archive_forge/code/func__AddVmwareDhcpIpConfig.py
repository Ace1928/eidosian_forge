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
def _AddVmwareDhcpIpConfig(ip_configuration_mutex_group, for_update=False):
    """Adds flags to specify DHCP configuration.

  Args:
    ip_configuration_mutex_group: The parent group to add the flag to.
    for_update: bool, True to add flags for update command, False to add flags
      for create command.
  """
    if for_update:
        return
    dhcp_config_group = ip_configuration_mutex_group.add_group(help='DHCP configuration group.')
    dhcp_config_group.add_argument('--enable-dhcp', help=textwrap.dedent('        Enable DHCP IP allocation for VMware user clusters.\n\n        While using DHCP, manual load balancing mode is not supported. For more details, see https://cloud.google.com/anthos/clusters/docs/on-prem/latest/how-to/manual-load-balance#setting_aside_node_ip_addresses.\n        '), action='store_true')