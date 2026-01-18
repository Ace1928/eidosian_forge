from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kms import maps
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util import parameter_info_lib
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
def AddRsaAesWrappedKeyFileFlag(parser, help_action):
    parser.add_argument('--rsa-aes-wrapped-key-file', help='Path to the wrapped RSA AES key file {}.'.format(help_action), hidden=True, action=actions.DeprecationAction('--rsa-aes-wrapped-key-file', warn='The {flag_name} flag is deprecated but will continue to be supported. Prefer to use the --wrapped-key-file flag instead.'))