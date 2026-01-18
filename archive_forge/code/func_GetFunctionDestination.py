from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.eventarc import triggers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.eventarc import flags
from googlecloudsdk.command_lib.eventarc import types
from googlecloudsdk.core import log
def GetFunctionDestination(self, args, old_trigger):
    if args.IsSpecified('destination_function'):
        return args.destination_function
    if old_trigger.destination.cloudFunction:
        return old_trigger.destination.cloudFunction.split('/')[5]
    raise exceptions.InvalidArgumentException('--destination-function-location', 'The specified trigger is not for a function destination.')