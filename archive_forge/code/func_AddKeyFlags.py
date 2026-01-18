from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.core.util import files
def AddKeyFlags(parser, action, additional_help=''):
    """Adds --key and --key-file flags to oslogin commands."""
    key_arg = parser.add_mutually_exclusive_group(required=True)
    key_arg.add_argument('--key', help='          The SSH public key to {0} the OS Login Profile.{1}\n          '.format(action, additional_help))
    key_arg.add_argument('--key-file', help='          The path to a file containing an SSH public key to {0} the\n          OS Login Profile.{1}\n          '.format(action, additional_help))