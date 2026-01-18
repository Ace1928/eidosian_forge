from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddCPUCount(parser, required=True):
    """Adds a --cpu-count flag to parser.

  Args:
    parser: argparse.Parser: Parser object for command line inputs.
    required: Whether or not --cpu-count is required.
  """
    parser.add_argument('--cpu-count', required=required, type=int, choices=[2, 4, 8, 16, 32, 64, 96, 128], help='Whole number value indicating how many vCPUs the machine should contain. Each vCPU count corresponds to a N2 high-mem machine: (https://cloud.google.com/compute/docs/general-purpose-machines#n2_machines).')