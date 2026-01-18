from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
class Poller(waiter.OperationPoller):
    """Poller for synchronous patch job execution."""

    def __init__(self, client, messages):
        """Initializes poller for patch job execution.

    Args:
      client: API client of the OsConfig service.
      messages: API messages of the OsConfig service.
    """
        self.client = client
        self.messages = messages
        self.patch_job_terminal_states = [self.messages.PatchJob.StateValueValuesEnum.SUCCEEDED, self.messages.PatchJob.StateValueValuesEnum.COMPLETED_WITH_ERRORS, self.messages.PatchJob.StateValueValuesEnum.TIMED_OUT, self.messages.PatchJob.StateValueValuesEnum.CANCELED]

    def IsDone(self, patch_job):
        """Overrides."""
        return patch_job.state in self.patch_job_terminal_states

    def Poll(self, request):
        """Overrides."""
        return self.client.projects_patchJobs.Get(request)

    def GetResult(self, patch_job):
        """Overrides."""
        return patch_job