from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddAcceleratorFields(parser):
    """Adds the --accelerator-type and --accelerator-count flags to the given parser."""
    group = parser.add_group(help='Accelerator settings')
    help_text = '  The type of accelerator resource to attach to the instance, for example,\n  "nvidia-tesla-p100".\n  '
    group.add_argument('--accelerator-type', type=str, help=help_text)
    help_text = '  The number of accelerator cards exposed to the instance.\n  '
    group.add_argument('--accelerator-count', type=int, help=help_text, required=True)