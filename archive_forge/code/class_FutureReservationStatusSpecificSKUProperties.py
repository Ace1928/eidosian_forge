from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FutureReservationStatusSpecificSKUProperties(_messages.Message):
    """Properties to be set for the Future Reservation.

  Fields:
    sourceInstanceTemplateId: ID of the instance template used to populate the
      Future Reservation properties.
  """
    sourceInstanceTemplateId = _messages.StringField(1)