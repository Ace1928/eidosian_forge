from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddIamInstanceProfile(parser, kind='cluster', required=True):
    """Adds the --iam-instance-profile flag."""
    parser.add_argument('--iam-instance-profile', required=required, help='Name or ARN of the IAM instance profile associated with the {}.'.format(kind))