from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.pubsub import subscriptions
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.pubsub import resource_args
from googlecloudsdk.command_lib.pubsub import util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def AddAckIdFlag(parser, action, add_deprecated=False):
    """Adds parsing and help text for ack_id flag."""
    help_text = 'One or more ACK_IDs to {} An ACK_ID is a [string that is returned to subscribers](https://cloud.google.com/pubsub/docs/reference/rpc/google.pubsub.v1#google.pubsub.v1.ReceivedMessage). along with the message. The ACK_ID is different from the [message ID](https://cloud.google.com/pubsub/docs/reference/rpc/google.pubsub.v1#google.pubsub.v1.PubsubMessage).'.format(action)
    group = parser
    if add_deprecated:
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('ack_id', nargs='*', help=help_text, action=actions.DeprecationAction('ACK_ID', show_message=lambda _: False, warn=DEPRECATION_FORMAT_STR.format('ACK_ID', '--ack-ids')))
    group.add_argument('--ack-ids', metavar='ACK_ID', required=not add_deprecated, type=arg_parsers.ArgList(), help=help_text)