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
def _AddPodAddressCidrBlocks(vmware_network_config_group, for_update=False):
    """Adds a flag to specify the IPv4 address ranges used in the pods in the cluster.

  Args:
    vmware_network_config_group: The parent group to add the flag to.
    for_update: bool, True to add flags for update command, False to add flags
      for create command.
  """
    required = False if for_update else True
    if for_update:
        return
    vmware_network_config_group.add_argument('--pod-address-cidr-blocks', metavar='POD_ADDRESS', type=arg_parsers.ArgList(min_length=1, max_length=1), required=required, help='IPv4 address range for all pods in the cluster.')