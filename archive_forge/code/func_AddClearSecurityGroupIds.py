from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddClearSecurityGroupIds(parser, noun):
    """Adds flag for clearing the security groups.

  Args:
    parser: The argparse.parser to add the arguments to.
    noun: The resource type to which the flag is applicable.
  """
    parser.add_argument('--clear-security-group-ids', action='store_true', default=None, help="Clear any additional security groups associated with the {}'s nodes. This does not remove the default security groups.".format(noun))