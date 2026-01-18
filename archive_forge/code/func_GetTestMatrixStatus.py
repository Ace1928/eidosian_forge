from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import time
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import exceptions
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
def GetTestMatrixStatus(self):
    """Fetch the response from the GetTestMatrix rpc.

    Returns:
      A TestMatrix message holding the current state of the created tests.

    Raises:
      HttpException if the Test service reports a backend error.
    """
    request = self._messages.TestingProjectsTestMatricesGetRequest(projectId=self._project, testMatrixId=self.matrix_id)
    try:
        return self._client.projects_testMatrices.Get(request)
    except apitools_exceptions.HttpError as e:
        exc = calliope_exceptions.HttpException(e)
        exc.error_format = 'Http error {status_code} while monitoring test run: {message}'
        raise exc