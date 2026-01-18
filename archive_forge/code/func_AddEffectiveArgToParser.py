from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddEffectiveArgToParser(parser, message):
    """Adds argument for the effective option.

  Args:
    parser: ArgumentInterceptor, An argparse parser.
    message: String, help text for flag.
  """
    parser.add_argument('--effective', action='store_true', required=False, help=message)