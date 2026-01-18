from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2CancelExecutionRequest(_messages.Message):
    """Request message for deleting an Execution.

  Fields:
    etag: A system-generated fingerprint for this version of the resource.
      This may be used to detect modification conflict during updates.
    validateOnly: Indicates that the request should be validated without
      actually cancelling any resources.
  """
    etag = _messages.StringField(1)
    validateOnly = _messages.BooleanField(2)