from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunProjectsLocationsJobsCreateRequest(_messages.Message):
    """A RunProjectsLocationsJobsCreateRequest object.

  Fields:
    googleCloudRunV2Job: A GoogleCloudRunV2Job resource to be passed as the
      request body.
    jobId: Required. The unique identifier for the Job. The name of the job
      becomes {parent}/jobs/{job_id}.
    parent: Required. The location and project in which this Job should be
      created. Format: projects/{project}/locations/{location}, where
      {project} can be project id or number.
    validateOnly: Indicates that the request should be validated and default
      values populated, without persisting the request or creating any
      resources.
  """
    googleCloudRunV2Job = _messages.MessageField('GoogleCloudRunV2Job', 1)
    jobId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    validateOnly = _messages.BooleanField(4)