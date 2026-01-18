from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
def _ListSteps(self, page_token):
    """Lists one page of steps using the ToolResults service.

    Args:
      page_token: A page token to attach to the List request. If it's None, then
        it returns at most the first 200 steps.

    Returns:
      A ListStepsResponse containing a single page's steps.

    Raises:
      HttpException if the ToolResults service reports a back-end error.
    """
    request = self._messages.ToolresultsProjectsHistoriesExecutionsStepsListRequest(projectId=self._project, historyId=self._history_id, executionId=self._execution_id, pageSize=100, pageToken=page_token)
    try:
        return self._client.projects_histories_executions_steps.List(request)
    except apitools_exceptions.HttpError as error:
        msg = 'Http error while listing test results of steps: ' + util.GetError(error)
        raise exceptions.HttpException(msg)