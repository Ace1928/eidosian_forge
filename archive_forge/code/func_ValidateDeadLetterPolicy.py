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
def ValidateDeadLetterPolicy(args):
    """Raises an exception if args has invalid dead letter arguments.

  Args:
    args (argparse.Namespace): Parsed arguments

  Raises:
    RequiredArgumentException: if max_delivery_attempts is set without
      dead_letter_topic being present.
  """
    if args.max_delivery_attempts and (not args.dead_letter_topic):
        raise exceptions.RequiredArgumentException('DEAD_LETTER_TOPIC', '--dead-letter-topic')