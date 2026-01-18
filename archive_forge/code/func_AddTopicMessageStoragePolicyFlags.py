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
def AddTopicMessageStoragePolicyFlags(parser, is_update):
    """Add flags for the Message Storage Policy.

  Args:
    parser: The argparse parser.
    is_update: Whether the operation is for updating message storage policy.
  """
    current_group = parser
    help_message = 'Options for explicitly specifying the [message storage policy](https://cloud.google.com/pubsub/docs/resource-location-restriction) for a topic.'
    if is_update:
        recompute_msp_group = current_group.add_group(mutex=True, help='Message storage policy options.')
        recompute_msp_group.add_argument('--recompute-message-storage-policy', action='store_true', help='If given, Pub/Sub recomputes the regions where messages can be stored at rest, based on your organization\'s "Resource  Location Restriction" policy.')
        current_group = recompute_msp_group
        help_message = f'{help_message} These fields can be set only if the `--recompute-message-storage-policy` flag is not set.'
    explicit_msp_group = current_group.add_argument_group(help=help_message)
    explicit_msp_group.add_argument('--message-storage-policy-allowed-regions', metavar='REGION', type=arg_parsers.ArgList(), required=True, help='A list of one or more Cloud regions where messages are allowed to be stored at rest.')
    explicit_msp_group.add_argument('--message-storage-policy-enforce-in-transit', action='store_true', help='Whether or not to enforce in-transit guarantees for this topic using the allowed regions. This ensures that publishing, pulling, and push delivery are only handled in allowed Cloud regions.')