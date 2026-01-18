from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlInstancesAcquireSsrsLeaseResponse(_messages.Message):
    """Acquire SSRS lease response.

  Fields:
    operationId: The unique identifier for this operation.
  """
    operationId = _messages.StringField(1)