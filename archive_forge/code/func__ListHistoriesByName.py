from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
def _ListHistoriesByName(self, history_name, page_size):
    """Lists histories by name using the Tool Results API.

    Args:
       history_name: string containing the history name.
       page_size: maximum number of histories to return.

    Returns:
      A list of histories matching the name.

    Raises:
      HttpException if the Tool Results service reports a backend error.
    """
    request = self._messages.ToolresultsProjectsHistoriesListRequest(projectId=self._project, filterByName=history_name, pageSize=page_size)
    try:
        response = self._client.projects_histories.List(request)
        log.debug('\nToolResultsHistories.List response:\n{0}\n'.format(response))
        return response
    except apitools_exceptions.HttpError as error:
        msg = 'Http error while getting list of Tool Results Histories:\n{0}'.format(util.GetError(error))
        raise exceptions.HttpException(msg)