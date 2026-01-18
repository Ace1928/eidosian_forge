from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GetGuestAttributesResponse(_messages.Message):
    """Response for GetGuestAttributes.

  Fields:
    guestAttributes: The guest attributes for the TPU workers.
  """
    guestAttributes = _messages.MessageField('GuestAttributes', 1, repeated=True)