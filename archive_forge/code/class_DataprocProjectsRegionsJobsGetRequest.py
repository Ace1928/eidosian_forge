from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsRegionsJobsGetRequest(_messages.Message):
    """A DataprocProjectsRegionsJobsGetRequest object.

  Fields:
    jobId: Required. The job ID.
    projectId: Required. The ID of the Google Cloud Platform project that the
      job belongs to.
    region: Required. The Dataproc region in which to handle the request.
  """
    jobId = _messages.StringField(1, required=True)
    projectId = _messages.StringField(2, required=True)
    region = _messages.StringField(3, required=True)