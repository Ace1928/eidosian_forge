from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GetGuestAttributesRequest(_messages.Message):
    """Request for GetGuestAttributes.

  Fields:
    queryPath: The guest attributes path to be queried.
    workerIds: The 0-based worker ID. If it is empty, all workers'
      GuestAttributes will be returned.
  """
    queryPath = _messages.StringField(1)
    workerIds = _messages.StringField(2, repeated=True)