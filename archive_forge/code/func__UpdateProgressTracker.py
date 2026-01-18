from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.os_config import utils as osconfig_api_utils
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.os_config import utils as osconfig_command_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_projector
import six
def _UpdateProgressTracker(tracker, patch_job, unused_status):
    """Updates the progress tracker on screen based on patch job details.

  Args:
    tracker: Progress tracker to be updated.
    patch_job: Patch job being executed.
  """
    details_json = resource_projector.MakeSerializable(patch_job.instanceDetailsSummary)
    if not details_json or details_json == '{}':
        if not tracker.IsRunning('pre-summary'):
            tracker.StartStage('pre-summary')
        else:
            tracker.UpdateStage('pre-summary', 'Please wait...')
    else:
        details_str = _CreateExecutionUpdateMessage(patch_job.percentComplete, details_json)
        if tracker.IsRunning('pre-summary'):
            tracker.CompleteStage('pre-summary', 'Done!')
            tracker.StartStage('with-summary')
            tracker.UpdateStage('with-summary', details_str)
        else:
            tracker.UpdateStage('with-summary', details_str)