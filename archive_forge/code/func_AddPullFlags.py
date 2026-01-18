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
def AddPullFlags(parser, add_deprecated=False, add_wait=False, add_return_immediately=False):
    """Adds the main set of message pulling flags to a parser."""
    if add_deprecated:
        parser.add_argument('--max-messages', type=int, default=1, help='The maximum number of messages that Cloud Pub/Sub can return in this response.', action=actions.DeprecationAction('--max-messages', warn='`{flag_name}` is deprecated. Please use --limit instead.'))
    AddBooleanFlag(parser=parser, flag_name='auto-ack', action='store_true', default=False, help_text='Automatically ACK every message pulled from this subscription.')
    if add_wait and add_return_immediately:
        parser = parser.add_group(mutex=True, help='Pull timeout behavior.')
    if add_wait:
        parser.add_argument('--wait', default=True, help='Wait (for a bounded amount of time) for new messages from the subscription, if there are none.', action=actions.DeprecationAction('--wait', warn='`{flag_name}` is deprecated. This flag is non-operational, as the wait behavior is now the default.', action='store_true'))
    if add_return_immediately:
        AddBooleanFlag(parser=parser, flag_name='return-immediately', action='store_true', default=False, help_text='If this flag is set, the system responds immediately with any messages readily available in memory buffers. If no messages are available in the buffers, returns an empty list of messages as response, even if having messages in the backlog. Do not set this flag as it adversely impacts the performance of pull.')