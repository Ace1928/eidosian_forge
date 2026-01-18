from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.domains import registrations
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddConfigureDNSSettingsFlagsToParser(parser):
    """Get flags for changing DNS settings.

  Args:
    parser: argparse parser to which to add these flags.
  """
    _AddDNSSettingsFlagsToParser(parser, mutation_op=MutationOp.UPDATE)
    base.Argument('--unsafe-dns-update', default=False, action='store_true', help='Use this flag to allow DNS changes that may make your domain stop serving.').AddToParser(parser)