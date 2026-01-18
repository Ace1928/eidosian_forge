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
def AddRegion(parser, required=True):
    """Adds a --region flag to parser.

  Args:
    parser: argparse.Parser: Parser object for command line inputs.
    required: Whether or not --region is required.
  """
    parser.add_argument('--region', required=required, type=str, help='Regional location (e.g. asia-east1, us-east1). See the full list of regions at https://cloud.google.com/sql/docs/instance-locations.')