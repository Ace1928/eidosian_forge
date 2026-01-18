from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ChangeQuorumMetadata(_messages.Message):
    """Metadata type for the long-running operation returned by ChangeQuorum.

  Fields:
    endTime: If set, the time at which this operation failed or was completed
      successfully.
    request: The request for ChangeQuorum.
    startTime: Time the request was received.
  """
    endTime = _messages.StringField(1)
    request = _messages.MessageField('ChangeQuorumRequest', 2)
    startTime = _messages.StringField(3)