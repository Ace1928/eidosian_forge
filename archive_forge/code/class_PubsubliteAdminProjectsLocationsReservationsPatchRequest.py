from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PubsubliteAdminProjectsLocationsReservationsPatchRequest(_messages.Message):
    """A PubsubliteAdminProjectsLocationsReservationsPatchRequest object.

  Fields:
    name: The name of the reservation. Structured like: projects/{project_numb
      er}/locations/{location}/reservations/{reservation_id}
    reservation: A Reservation resource to be passed as the request body.
    updateMask: Required. A mask specifying the reservation fields to change.
  """
    name = _messages.StringField(1, required=True)
    reservation = _messages.MessageField('Reservation', 2)
    updateMask = _messages.StringField(3)