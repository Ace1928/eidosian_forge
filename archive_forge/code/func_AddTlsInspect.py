from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddTlsInspect(parser):
    """Adds the option to turn on TLS decryption on the rule."""
    parser.add_argument('--tls-inspect', required=False, action=arg_parsers.StoreTrueFalseAction, help='Use this flag to indicate whether TLS traffic should be inspected using the TLS inspection policy when the security profile group is applied. Default: no TLS inspection.')