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
def ParseAckIdsArgs(args):
    """Gets the list of ack_ids from args.

  This is only necessary because we are deprecating the positional `ack_id`.
  Putting the positional and a flag in an argument group, will group flag
  under positional args. This would be confusing.

  DeprecationAction can't be used here because in order to make the positional
  argument optional, we have to use `nargs='*'`. Since this means zero or more,
  the DeprecationAction warn message is always triggered.

  This function does not exist in util.py in order to group the explanation for
  why this exists with the deprecated flags.

  Once the positional is removed, this function can be removed.

  Args:
    args (argparse.Namespace): Parsed arguments

  Returns:
    list[str]: List of ack ids.
  """
    if args.ack_id:
        log.warning(DEPRECATION_FORMAT_STR.format('ACK_ID', '--ack-ids'))
    ack_ids = args.ack_id or args.ack_ids
    if not isinstance(ack_ids, list):
        ack_ids = [ack_ids]
    return ack_ids