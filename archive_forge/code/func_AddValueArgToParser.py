from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.resource_manager import completers
from googlecloudsdk.command_lib.util.args import common_args
def AddValueArgToParser(parser):
    """Adds argument for a list of values to the parser.

  Args:
    parser: ArgumentInterceptor, An argparse parser.
  """
    parser.add_argument('value', metavar='VALUE', nargs='*', help='Values to add to the policy. The set of valid values corresponding to the different constraints are covered here: https://cloud.google.com/resource-manager/docs/organization-policy/org-policy-constraints')