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
def AddSetSuperuser(parser):
    """Adds a --superuser flag to parser.

  Args:
    parser: argparse.Parser: Parser object for command line inputs.
  """
    parser.add_argument('--superuser', type=arg_parsers.ArgBoolean(), default=False, required=True, help='If true, user will have AlloyDB superuser privileges')