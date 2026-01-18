from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlInstancesReleaseSsrsLeaseResponse(_messages.Message):
    """The response for the release of the SSRS lease.

  Fields:
    operationId: The operation ID.
  """
    operationId = _messages.StringField(1)