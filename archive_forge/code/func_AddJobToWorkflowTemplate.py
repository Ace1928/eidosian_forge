from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.util.args import labels_util
def AddJobToWorkflowTemplate(args, dataproc, ordered_job):
    """Add an ordered job to the workflow template."""
    template = args.CONCEPTS.workflow_template.Parse()
    workflow_template = dataproc.GetRegionsWorkflowTemplate(template, args.version)
    jobs = workflow_template.jobs if workflow_template.jobs is not None else []
    jobs.append(ordered_job)
    workflow_template.jobs = jobs
    response = dataproc.client.projects_regions_workflowTemplates.Update(workflow_template)
    return response