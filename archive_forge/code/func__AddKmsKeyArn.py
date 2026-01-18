from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.util.apis import arg_utils
def _AddKmsKeyArn(parser, prefix, target, required=False):
    parser.add_argument('--{}-kms-key-arn'.format(prefix), required=required, help='Amazon Resource Name (ARN) of the AWS KMS key to encrypt the {}.'.format(target))