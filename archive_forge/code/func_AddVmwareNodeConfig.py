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
def AddVmwareNodeConfig(parser: parser_arguments.ArgumentInterceptor, for_update: bool=False, release_track: base.ReleaseTrack=None):
    """Adds flags to specify the configuration of the node pool.

  Args:
    parser: The argparse parser to add the flag to.
    for_update: bool, True to add flags for update command, False to add flags
      for create command.
    release_track: The release track of this command.
  """
    vmware_node_config_group = parser.add_group(help='Configuration of the node pool', required=False if for_update else True)
    _AddCpus(vmware_node_config_group)
    _AddMemoryMb(vmware_node_config_group)
    _AddReplicas(vmware_node_config_group)
    _AddImageType(vmware_node_config_group, for_update=for_update)
    _AddImage(vmware_node_config_group)
    _AddBootDiskSizeGb(vmware_node_config_group)
    _AddNodeTaint(vmware_node_config_group)
    _AddNodeLabels(vmware_node_config_group)
    _AddVmwareVsphereConfig(vmware_node_config_group, release_track=release_track, for_update=for_update)
    _AddEnableLoadBalancer(vmware_node_config_group, for_update=for_update)