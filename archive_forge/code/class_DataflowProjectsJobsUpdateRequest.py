from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowProjectsJobsUpdateRequest(_messages.Message):
    """A DataflowProjectsJobsUpdateRequest object.

  Fields:
    job: A Job resource to be passed as the request body.
    jobId: The job ID.
    location: The [regional endpoint]
      (https://cloud.google.com/dataflow/docs/concepts/regional-endpoints)
      that contains this job.
    projectId: The ID of the Cloud Platform project that the job belongs to.
    updateMask: The list of fields to update relative to Job. If empty, only
      RequestedJobState will be considered for update. If the FieldMask is not
      empty and RequestedJobState is none/empty, The fields specified in the
      update mask will be the only ones considered for update. If both
      RequestedJobState and update_mask are specified, an error will be
      returned as we cannot update both state and mask.
  """
    job = _messages.MessageField('Job', 1)
    jobId = _messages.StringField(2, required=True)
    location = _messages.StringField(3)
    projectId = _messages.StringField(4, required=True)
    updateMask = _messages.StringField(5)