from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddEncryptedInterconnectRouter(parser):
    """Adds encrypted interconnect router flag."""
    parser.add_argument('--encrypted-interconnect-router', required=False, action='store_true', default=None, help='Indicates if a router is dedicated for use with encrypted interconnect attachments (VLAN attachments).')