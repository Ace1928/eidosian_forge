from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataflow import apis
from googlecloudsdk.core.util import times
@staticmethod
def _JobTypeForJob(job_type):
    """Return a string describing the job type.

    Args:
      job_type: The job type enum
    Returns:
      string describing the job type
    """
    type_value_enum = apis.GetMessagesModule().Job.TypeValueValuesEnum
    value_map = {type_value_enum.JOB_TYPE_BATCH: 'Batch', type_value_enum.JOB_TYPE_STREAMING: 'Streaming'}
    return value_map.get(job_type, 'Unknown')