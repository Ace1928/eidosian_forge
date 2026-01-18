from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.util.args import labels_util
def CreateWorkflowTemplateOrderedJob(args, dataproc):
    """Create an ordered job for workflow template."""
    ordered_job = dataproc.messages.OrderedJob(stepId=args.step_id)
    if args.start_after:
        ordered_job.prerequisiteStepIds = args.start_after
    return ordered_job