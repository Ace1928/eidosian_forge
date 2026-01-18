from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SubmitJobRequest(_messages.Message):
    """A request to submit a job.

  Fields:
    job: Required. The job resource.
    requestId: Optional. A unique id used to identify the request. If the
      server receives two SubmitJobRequest (https://cloud.google.com/dataproc/
      docs/reference/rpc/google.cloud.dataproc.v1#google.cloud.dataproc.v1.Sub
      mitJobRequest)s with the same id, then the second request will be
      ignored and the first Job created and stored in the backend is
      returned.It is recommended to always set this value to a UUID
      (https://en.wikipedia.org/wiki/Universally_unique_identifier).The id
      must contain only letters (a-z, A-Z), numbers (0-9), underscores (_),
      and hyphens (-). The maximum length is 40 characters.
  """
    job = _messages.MessageField('Job', 1)
    requestId = _messages.StringField(2)