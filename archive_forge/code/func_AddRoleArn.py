from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddRoleArn(parser, required=True):
    parser.add_argument('--role-arn', required=required, help='Amazon Resource Name (ARN) of the IAM role to assume when managing AWS resources.')