from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MlProjectsJobsCancelRequest(_messages.Message):
    """A MlProjectsJobsCancelRequest object.

  Fields:
    googleCloudMlV1CancelJobRequest: A GoogleCloudMlV1CancelJobRequest
      resource to be passed as the request body.
    name: Required. The name of the job to cancel.
  """
    googleCloudMlV1CancelJobRequest = _messages.MessageField('GoogleCloudMlV1CancelJobRequest', 1)
    name = _messages.StringField(2, required=True)