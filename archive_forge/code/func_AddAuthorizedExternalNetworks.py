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
def AddAuthorizedExternalNetworks(parser):
    """Adds a `--authorized-external-networks` flag to parser.

  Args:
    parser: argparse.Parser: Parser object for command line inputs.
  """
    parser.add_argument('--authorized-external-networks', type=arg_parsers.ArgList(), metavar='AUTHORIZED_NETWORK', required=False, help='Comma-separated list of authorized external networks to set on the instance. Authorized networks should use CIDR notation (e.g. 1.2.3.4/30). This flag is only allowed to be set for instances with public IP enabled.')