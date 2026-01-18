from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataflow import apis
from googlecloudsdk.api_lib.dataflow import job_display
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dataflow import dataflow_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_filter
from googlecloudsdk.core.util import times
def _JobSummariesForProject(self, project_id, args, filter_predicate):
    """Get the list of job summaries that match the predicate.

    Args:
      project_id: The project ID to retrieve
      args: parsed command line arguments
      filter_predicate: The filter predicate to apply

    Returns:
      An iterator over all the matching jobs.
    """
    request = None
    service = None
    status_filter = self._StatusArgToFilter(args.status, args.region)
    if args.region:
        request = apis.Jobs.LIST_REQUEST(projectId=project_id, location=args.region, filter=status_filter)
        service = apis.Jobs.GetService()
    else:
        log.status.Print('`--region` not set; getting jobs from all available regions. ' + 'Some jobs may be missing in the event of an outage. ' + 'https://cloud.google.com/dataflow/docs/concepts/regional-endpoints')
        request = apis.Jobs.AGGREGATED_LIST_REQUEST(projectId=project_id, filter=status_filter)
        service = apis.GetClientInstance().projects_jobs
    return dataflow_util.YieldFromList(project_id=project_id, region_id=args.region, service=service, request=request, limit=args.limit, batch_size=args.page_size, field='jobs', batch_size_attribute='pageSize', predicate=filter_predicate)