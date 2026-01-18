from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddKeepaliveIntervalArg(parser):
    """Adds keepalive interval argument for routers."""
    parser.add_argument('--keepalive-interval', type=arg_parsers.Duration(default_unit='s', lower_bound='20s', upper_bound='60s'), help="The interval between BGP keepalive messages that are sent to the peer. If set, this value must be between 20 and 60 seconds. The default is 20 seconds. See $ gcloud topic datetimes for information on duration formats.\n\nBGP systems exchange keepalive messages to determine whether a link or host has failed or is no longer available. Hold time is the length of time in seconds that the BGP session is considered operational without any activity. After the hold time expires, the session is dropped.\n\nHold time is three times the interval at which keepalive messages are sent, and the hold time is the maximum number of seconds allowed to elapse between successive keepalive messages that BGP receives from a peer. BGP will use the smaller of either the local hold time value or the peer's  hold time value as the hold time for the BGP connection between the two peers.")