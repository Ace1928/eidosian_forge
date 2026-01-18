from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddConnectionPreference(parser):
    """Add support for --connection-preference flag."""
    parser.add_argument('--connection-preference', choices=['ACCEPT_AUTOMATIC', 'ACCEPT_MANUAL'], type=lambda x: x.replace('-', '_').upper(), default='ACCEPT_AUTOMATIC', help='      The connection preference of network attachment.\n      The value can be set to ACCEPT_AUTOMATIC or ACCEPT_MANUAL.\n      An ACCEPT_AUTOMATIC network attachment is one that\n      always accepts the connection from producer NIC.\n      An ACCEPT_MANUAL network attachment is one that\n      requires an explicit addition of the producer project id\n      or project number to the producer accept list.\n      ')