from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.resource_manager import completers
from googlecloudsdk.command_lib.util.args import common_args
def AddCustomConstraintArgToParser(parser):
    """Adds argument for the custom constraint name to the parser.

  Args:
    parser: ArgumentInterceptor, An argparse parser.
  """
    parser.add_argument('custom_constraint', metavar='CUSTOM_CONSTRAINT', help='Name of the custom constraint.')