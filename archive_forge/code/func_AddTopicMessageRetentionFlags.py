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
def AddTopicMessageRetentionFlags(parser, is_update):
    """Add flags for the topic message retention property to the parser.

  Args:
    parser: The argparse parser.
    is_update: Whether the operation is for updating message retention.
  """
    current_group = parser
    if is_update:
        mutual_exclusive_group = parser.add_mutually_exclusive_group()
        AddBooleanFlag(parser=mutual_exclusive_group, flag_name='clear-message-retention-duration', action='store_true', default=None, help_text='If set, clear the message retention duration from the topic.')
        current_group = mutual_exclusive_group
    current_group.add_argument('--message-retention-duration', type=arg_parsers.Duration(lower_bound='10m', upper_bound='31d'), help='          Indicates the minimum duration to retain a message after it is\n          published to the topic. If this field is set, messages published to\n          the topic in the last MESSAGE_RETENTION_DURATION are always available\n          to subscribers. For instance, it allows any attached subscription to\n          seek to a timestamp that is up to MESSAGE_RETENTION_DURATION in the\n          past. If this field is not set, message retention is controlled by\n          settings on individual subscriptions. The minimum is 10 minutes and\n          the maximum is 31 days. {}'.format(DURATION_HELP_STR))