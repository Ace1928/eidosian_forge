from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util import completers
def AddPasswordPolicyComplexity(parser, hidden=False):
    """Add the flag to specify password policy complexity.

  Args:
    parser: The current argparse parser to add this to.
    hidden: if the field needs to be hidden.
  """
    parser.add_argument('--password-policy-complexity', choices={'COMPLEXITY_UNSPECIFIED': 'The default value if COMPLEXITY_DEFAULT is not specified. It implies that complexity check is not enabled.', 'COMPLEXITY_DEFAULT': 'A combination of lowercase, uppercase, numeric, and non-alphanumeric characters.'}, required=False, default=None, help='The complexity of the password. This flag is available only for PostgreSQL.', hidden=hidden)