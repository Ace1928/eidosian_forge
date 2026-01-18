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
def AddAvailabilityType(parser):
    """Adds an '--availability-type' flag to the parser.

  Args:
    parser: argparse.Parser: Parser object for command line inputs.
  """
    choices_arg = {'REGIONAL': 'Provide high availability instances. Recommended for production instances; instances automatically fail over to another zone within your selected region.', 'ZONAL': 'Provide zonal availability instances. Not recommended for production instances; instance does not automatically fail over to another zone.'}
    parser.add_argument('--availability-type', required=False, choices=choices_arg, help='Specifies level of availability.')