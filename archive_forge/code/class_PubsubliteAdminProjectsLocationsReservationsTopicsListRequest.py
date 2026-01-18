from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PubsubliteAdminProjectsLocationsReservationsTopicsListRequest(_messages.Message):
    """A PubsubliteAdminProjectsLocationsReservationsTopicsListRequest object.

  Fields:
    name: Required. The name of the reservation whose topics to list.
      Structured like: projects/{project_number}/locations/{location}/reservat
      ions/{reservation_id}
    pageSize: The maximum number of topics to return. The service may return
      fewer than this value. If unset or zero, all topics for the given
      reservation will be returned.
    pageToken: A page token, received from a previous `ListReservationTopics`
      call. Provide this to retrieve the subsequent page. When paginating, all
      other parameters provided to `ListReservationTopics` must match the call
      that provided the page token.
  """
    name = _messages.StringField(1, required=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)