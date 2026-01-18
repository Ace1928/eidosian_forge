from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddEnableLogging(parser, required=False):
    """Adds the option to enable logging."""
    parser.add_argument('--enable-logging', required=required, action=arg_parsers.StoreTrueFalseAction, help='Use this flag to enable logging of connections that allowed or denied by this rule.')