from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.compute.resource_policies import flags as maintenance_flags
from googlecloudsdk.command_lib.util.args import labels_util
def AddDistributionTargetShapeArgs(parser):
    """Adds bulk creation target shape arguments to parser."""
    choices_text = {'ANY_SINGLE_ZONE': 'Enforces VM placement in one allowed zone. Use this to avoid cross-zone network egress or to reduce network latency. This is the default value.', 'BALANCED': 'Allows distribution of VMs in zones where resources are available while distributing VMs as evenly as possible across selected zones to minimize the impact of zonal failures. Recommended for highly available serving or batch workloads.', 'ANY': 'Allows creating VMs in multiple zones if one zone cannot accommodate all the requested VMs. The resulting distribution shapes can vary.'}
    parser.add_argument('--target-distribution-shape', metavar='SHAPE', type=lambda x: x.upper(), choices=choices_text, help='\n        Specifies whether and how to distribute VMs across multiple zones in a\n        region or to enforce placement of VMs in a single zone.\n        The default shape is `ANY_SINGLE_ZONE`.\n      ')