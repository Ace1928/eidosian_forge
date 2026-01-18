from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PubsubliteAdminProjectsLocationsReservationsDeleteRequest(_messages.Message):
    """A PubsubliteAdminProjectsLocationsReservationsDeleteRequest object.

  Fields:
    name: Required. The name of the reservation to delete. Structured like: pr
      ojects/{project_number}/locations/{location}/reservations/{reservation_i
      d}
  """
    name = _messages.StringField(1, required=True)