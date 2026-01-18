from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsconfigProjectsPatchJobsExecuteRequest(_messages.Message):
    """A OsconfigProjectsPatchJobsExecuteRequest object.

  Fields:
    executePatchJobRequest: A ExecutePatchJobRequest resource to be passed as
      the request body.
    parent: Required. The project in which to run this patch in the form
      `projects/*`
  """
    executePatchJobRequest = _messages.MessageField('ExecutePatchJobRequest', 1)
    parent = _messages.StringField(2, required=True)