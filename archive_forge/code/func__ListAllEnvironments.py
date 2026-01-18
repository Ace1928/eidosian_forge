from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
def _ListAllEnvironments(self):
    """Lists all environments of a test execution using the ToolResults service.

    Returns:
      A ListEnvironmentsResponse containing all environments within execution.
    """
    response = self._ListEnvironments(None)
    environments = []
    environments.extend(response.environments)
    while response.nextPageToken:
        response = self._ListEnvironments(response.nextPageToken)
        environments.extend(response.environments)
    return environments