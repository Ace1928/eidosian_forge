from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import transfer
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def _GetDefaultBucket(self, tr_client, tr_messages):
    """Fetch the project's default GCS bucket name for storing tool results."""
    request = tr_messages.ToolresultsProjectsInitializeSettingsRequest(projectId=self._project)
    try:
        response = tr_client.projects.InitializeSettings(request)
        return response.defaultBucket
    except apitools_exceptions.HttpError as error:
        code, err_msg = util.GetErrorCodeAndMessage(error)
        if code == HTTP_FORBIDDEN:
            msg = 'Permission denied while fetching the default results bucket (Error {0}: {1}). Is billing enabled for project: [{2}]?'.format(code, err_msg, self._project)
        else:
            msg = 'Http error while trying to fetch the default results bucket:\nResponseError {0}: {1}'.format(code, err_msg)
        raise exceptions.HttpException(msg)