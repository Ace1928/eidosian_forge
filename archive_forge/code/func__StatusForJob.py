from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataflow import apis
from googlecloudsdk.core.util import times
@staticmethod
def _StatusForJob(job_state):
    """Return a string describing the job state.

    Args:
      job_state: The job state enum
    Returns:
      string describing the job state
    """
    state_value_enum = apis.GetMessagesModule().Job.CurrentStateValueValuesEnum
    value_map = {state_value_enum.JOB_STATE_CANCELLED: 'Cancelled', state_value_enum.JOB_STATE_CANCELLING: 'Cancelling', state_value_enum.JOB_STATE_DONE: 'Done', state_value_enum.JOB_STATE_DRAINED: 'Drained', state_value_enum.JOB_STATE_DRAINING: 'Draining', state_value_enum.JOB_STATE_FAILED: 'Failed', state_value_enum.JOB_STATE_PENDING: 'Pending', state_value_enum.JOB_STATE_QUEUED: 'Queued', state_value_enum.JOB_STATE_RUNNING: 'Running', state_value_enum.JOB_STATE_STOPPED: 'Stopped', state_value_enum.JOB_STATE_UPDATED: 'Updated'}
    return value_map.get(job_state, 'Unknown')