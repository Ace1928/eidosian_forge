from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.compute.networks import flags as compute_network_flags
from googlecloudsdk.command_lib.compute.networks.subnets import flags as compute_subnet_flags
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.notebooks import completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddCreateRuntimeFlags(api_version, parser):
    """Construct groups and arguments specific to the runtime creation."""
    AddRuntimeResource(api_version, parser)
    runtime_type_group = parser.add_group(mutex=True, required=True)
    runtime_type_group.add_argument('--runtime-type', help='runtime type')
    machine_type_group = runtime_type_group.add_group()
    machine_type_group.add_argument('--machine-type', help='machine type', required=True)
    local_disk_group = machine_type_group.add_group()
    local_disk_group.add_argument('--interface', help='runtime interface')
    local_disk_group.add_argument('--source', help='runtime source')
    local_disk_group.add_argument('--mode', help='runtime mode')
    local_disk_group.add_argument('--type', help='runtime type')
    access_config_group = parser.add_group(required=True)
    access_config_group.add_argument('--runtime-access-type', help='access type')
    access_config_group.add_argument('--runtime-owner', help='runtime owner')
    software_config_group = parser.add_group()
    software_config_group.add_argument('--idle-shutdown-timeout', help='idle shutdown timeout')
    software_config_group.add_argument('--install-gpu-driver', help='install gpu driver')
    software_config_group.add_argument('--custom-gpu-driver-path', help='custom gpu driver path')
    software_config_group.add_argument('--post-startup-script', help='post startup script')
    software_config_group.add_argument('--post-startup-script-behavior', help='post startup script behavior')