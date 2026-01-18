from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
class JobsClient(object):
    """Client used for interacting with job related service from the Data Pipelines API."""

    def __init__(self, client=None, messages=None):
        self.client = client or GetClientInstance()
        self.messages = messages or GetMessagesModule()
        self._service = self.client.projects_locations_pipelines_jobs

    def List(self, limit=None, page_size=50, pipeline=''):
        """Make API calls to list jobs for pipelines.

    Args:
      limit: int or None, the total number of results to return.
      page_size: int, the number of entries in each batch (affects requests
        made, but not the yielded results).
      pipeline: string, the name of the pipeline to list jobs for.

    Returns:
      Generator that yields jobs.
    """
        list_req = self.messages.DatapipelinesProjectsLocationsPipelinesJobsListRequest(parent=pipeline)
        return list_pager.YieldFromList(self._service, list_req, field='jobs', method='List', batch_size=page_size, limit=limit, batch_size_attribute='pageSize')