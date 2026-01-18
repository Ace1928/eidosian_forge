from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsRegionsJobsSubmitRequest(_messages.Message):
    """A DataprocProjectsRegionsJobsSubmitRequest object.

  Fields:
    projectId: Required. The ID of the Google Cloud Platform project that the
      job belongs to.
    region: Required. The Dataproc region in which to handle the request.
    submitJobRequest: A SubmitJobRequest resource to be passed as the request
      body.
  """
    projectId = _messages.StringField(1, required=True)
    region = _messages.StringField(2, required=True)
    submitJobRequest = _messages.MessageField('SubmitJobRequest', 3)