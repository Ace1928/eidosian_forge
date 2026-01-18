from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchProjectsLocationsJobsCreateRequest(_messages.Message):
    """A BatchProjectsLocationsJobsCreateRequest object.

  Fields:
    job: A Job resource to be passed as the request body.
    jobId: ID used to uniquely identify the Job within its parent scope. This
      field should contain at most 63 characters and must start with lowercase
      characters. Only lowercase characters, numbers and '-' are accepted. The
      '-' character cannot be the first or the last one. A system generated ID
      will be used if the field is not set. The job.name field in the request
      will be ignored and the created resource name of the Job will be
      "{parent}/jobs/{job_id}".
    parent: Required. The parent resource name where the Job will be created.
      Pattern: "projects/{project}/locations/{location}"
    requestId: Optional. An optional request ID to identify requests. Specify
      a unique request ID so that if you must retry your request, the server
      will know to ignore the request if it has already been completed. The
      server will guarantee that for at least 60 minutes since the first
      request. For example, consider a situation where you make an initial
      request and the request times out. If you make the request again with
      the same request ID, the server can check if original operation with the
      same request ID was received, and if so, will ignore the second request.
      This prevents clients from accidentally creating duplicate commitments.
      The request ID must be a valid UUID with the exception that zero UUID is
      not supported (00000000-0000-0000-0000-000000000000).
  """
    job = _messages.MessageField('Job', 1)
    jobId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    requestId = _messages.StringField(4)