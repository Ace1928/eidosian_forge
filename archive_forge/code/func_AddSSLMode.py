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
def AddSSLMode(parser, update=False):
    """Adds SSL Mode flag.

  Args:
    parser: argparse.Parser: Parser object for command line inputs.
    update: If True, does not set the default SSL mode.
  """
    ssl_mode_help = 'Specify the SSL mode to use when the instance connects to the database.'
    if update:
        parser.add_argument('--ssl-mode', required=False, type=str, choices={'ENCRYPTED_ONLY': 'SSL connections are required. CA verification is not enforced.', 'ALLOW_UNENCRYPTED_AND_ENCRYPTED': 'SSL connections are optional. CA verification is not enforced.'}, help=ssl_mode_help)
    else:
        ssl_mode_help += ' Default SSL mode is ENCRYPTED_ONLY.'
        parser.add_argument('--ssl-mode', required=False, type=str, choices={'ENCRYPTED_ONLY': 'SSL connections are required. CA verification is not enforced.', 'ALLOW_UNENCRYPTED_AND_ENCRYPTED': 'SSL connections are optional. CA verification is not enforced.'}, default='ENCRYPTED_ONLY', help=ssl_mode_help)