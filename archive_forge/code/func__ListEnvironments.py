from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
def _ListEnvironments(self, page_token):
    """Lists one page of environments using the ToolResults service.

    Args:
      page_token: A page token to attach to the List request. If it's None, then
        it returns a maximum of 200 Environments.

    Returns:
      A ListEnvironmentsResponse containing a single page's environments.

    Raises:
      HttpException if the ToolResults service reports a back-end error.
    """
    request = self._messages.ToolresultsProjectsHistoriesExecutionsEnvironmentsListRequest(projectId=self._project, historyId=self._history_id, executionId=self._execution_id, pageSize=100, pageToken=page_token)
    try:
        return self._client.projects_histories_executions_environments.List(request)
    except apitools_exceptions.HttpError as error:
        msg = 'Http error while listing test results: ' + util.GetError(error)
        raise exceptions.HttpException(msg)