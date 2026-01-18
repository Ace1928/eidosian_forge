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
def AddVCenterConfig(parser: parser_arguments.ArgumentInterceptor) -> None:
    """Adds vCenterConfig flags."""
    vcenter_config_group = parser.add_group(help='vCenter configurations for the cluster. If not specified, it is inherited from the admin cluster.')
    vcenter_config_group.add_argument('--vcenter-resource-pool', type=str, help='Name of the vCenter resource pool for the user cluster.')
    vcenter_config_group.add_argument('--vcenter-datastore', type=str, help='Name of the vCenter datastore for the user cluster.')
    vcenter_config_group.add_argument('--vcenter-datacenter', type=str, help='Name of the vCenter datacenter for the user cluster.')
    vcenter_config_group.add_argument('--vcenter-cluster', type=str, help='Name of the vCenter cluster for the user cluster.')
    vcenter_config_group.add_argument('--vcenter-folder', type=str, help='Name of the vCenter folder for the user cluster.')
    vcenter_config_group.add_argument('--vcenter-ca-cert-data', type=str, help='Name of the vCenter CA certificate public key for SSL verification.')
    vcenter_config_group.add_argument('--vcenter-storage-policy-name', type=str, help='Name of the vCenter storage policy for the user cluster.')