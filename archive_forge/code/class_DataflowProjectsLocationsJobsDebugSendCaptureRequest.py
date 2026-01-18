from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowProjectsLocationsJobsDebugSendCaptureRequest(_messages.Message):
    """A DataflowProjectsLocationsJobsDebugSendCaptureRequest object.

  Fields:
    jobId: The job id.
    location: The [regional endpoint]
      (https://cloud.google.com/dataflow/docs/concepts/regional-endpoints)
      that contains the job specified by job_id.
    projectId: The project id.
    sendDebugCaptureRequest: A SendDebugCaptureRequest resource to be passed
      as the request body.
  """
    jobId = _messages.StringField(1, required=True)
    location = _messages.StringField(2, required=True)
    projectId = _messages.StringField(3, required=True)
    sendDebugCaptureRequest = _messages.MessageField('SendDebugCaptureRequest', 4)