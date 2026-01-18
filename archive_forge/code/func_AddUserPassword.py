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
def AddUserPassword(parser, required=False):
    """Adds a --password flag to parser.

  Args:
    parser: argparse.Parser: Parser object for command line inputs.
    required: Whether or not --password is required.
  """
    parser.add_argument('--password', required=required, type=str, help='Password for this database user.')