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
def AddSeekFlags(parser):
    """Adds flags for the seek command to the parser."""
    seek_to_group = parser.add_mutually_exclusive_group(required=True)
    seek_to_group.add_argument('--time', type=arg_parsers.Datetime.Parse, help='          The time to seek to. Messages in the subscription that\n          were published before this time are marked as acknowledged, and\n          messages retained in the subscription that were published after\n          this time are marked as unacknowledged.\n          See $ gcloud topic datetimes for information on time formats.')
    seek_to_group.add_argument('--snapshot', help="The name of the snapshot. The snapshot's topic must be the same as that of the subscription.")
    parser.add_argument('--snapshot-project', help='          The name of the project the snapshot belongs to (if seeking to\n          a snapshot). If not set, it defaults to the currently selected\n          cloud project.')