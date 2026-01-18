from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PeeringZoneDeactivateResponse(_messages.Message):
    """A PeeringZoneDeactivateResponse object.

  Fields:
    deactivateSucceeded: True if the zone is deactivated by this request,
      false if the zone exists and is of type peering zone but was already
      deactivated.
    header: A ResponseHeader attribute.
  """
    deactivateSucceeded = _messages.BooleanField(1)
    header = _messages.MessageField('ResponseHeader', 2)