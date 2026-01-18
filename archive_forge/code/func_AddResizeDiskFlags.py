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
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.workbench import completers
from googlecloudsdk.core import properties
def AddResizeDiskFlags(parser):
    """Construct groups and arguments specific to the disk resizing."""
    AddInstanceResource(parser)
    disk_group = parser.add_group(help='Disk resizing configurations Amount needs to be greater than the existing size.', mutex=True, required=True)
    disk_group.add_argument('--boot-disk-size', type=int, help='Size of boot disk in GB attached to this instance, up to a maximum of 64000 GB (64 TB).')
    disk_group.add_argument('--data-disk-size', type=int, help='Size of data disk in GB attached to this instance, up to a maximum of 64000 GB (64 TB). ')