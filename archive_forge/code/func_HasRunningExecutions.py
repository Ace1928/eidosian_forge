from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.run import execution
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import connection_context
from googlecloudsdk.command_lib.run import deletion
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.command_lib.run import pretty_print
from googlecloudsdk.command_lib.run import resource_args
from googlecloudsdk.command_lib.run import serverless_operations
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
def HasRunningExecutions(self, job_ref, client):
    label_selector = '{label} = {name}, run.googleapis.com/servingState = Active'.format(label=execution.JOB_LABEL, name=job_ref.jobsId)
    for _ in client.ListExecutions(job_ref.Parent(), label_selector, limit=1, page_size=1):
        return True
    return False