from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.eventarc import triggers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.eventarc import flags
from googlecloudsdk.command_lib.eventarc import types
from googlecloudsdk.core import log
def GetWorkflowDestinationLocation(self, args, old_trigger):
    if args.IsSpecified('destination_workflow_location'):
        return args.destination_workflow_location
    if old_trigger.destination.workflow:
        return old_trigger.destination.workflow.split('/')[3]
    raise exceptions.InvalidArgumentException('--destination-workflow', 'The specified trigger is not for a workflow destination.')