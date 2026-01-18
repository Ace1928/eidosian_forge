from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
def _ListAllSteps(self):
    """Lists all steps for a test execution using the ToolResults service.

    Returns:
      The full list of steps for a test execution.
    """
    response = self._ListSteps(None)
    steps = []
    steps.extend(response.steps)
    while response.nextPageToken:
        response = self._ListSteps(response.nextPageToken)
        steps.extend(response.steps)
    return steps