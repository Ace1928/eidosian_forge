from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.core import yaml
def CheckJobComplete(self, name):
    """Returns a function to decide if log fetcher should continue polling.

    Args:
      name: String id of job.

    Returns:
      A one-argument function decides if log fetcher should continue.
    """
    request = self._messages.AiplatformProjectsLocationsHyperparameterTuningJobsGetRequest(name=name)
    response = self._service.Get(request)

    def ShouldContinue(periods_without_logs):
        if periods_without_logs <= 1:
            return True
        return response.endTime is None
    return ShouldContinue