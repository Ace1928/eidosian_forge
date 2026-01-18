from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StoragetransferTransferJobsDeleteRequest(_messages.Message):
    """A StoragetransferTransferJobsDeleteRequest object.

  Fields:
    jobName: Required. The job to delete.
    projectId: Required. The ID of the Google Cloud project that owns the job.
  """
    jobName = _messages.StringField(1, required=True)
    projectId = _messages.StringField(2, required=True)