from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsLakesEnvironmentsSessionsListRequest(_messages.Message):
    """A DataplexProjectsLocationsLakesEnvironmentsSessionsListRequest object.

  Fields:
    filter: Optional. Filter request. The following mode filter is supported
      to return only the sessions belonging to the requester when the mode is
      USER and return sessions of all the users when the mode is ADMIN. When
      no filter is sent default to USER mode. NOTE: When the mode is ADMIN,
      the requester should have dataplex.environments.listAllSessions
      permission to list all sessions, in absence of the permission, the
      request fails.mode = ADMIN | USER
    pageSize: Optional. Maximum number of sessions to return. The service may
      return fewer than this value. If unspecified, at most 10 sessions will
      be returned. The maximum value is 1000; values above 1000 will be
      coerced to 1000.
    pageToken: Optional. Page token received from a previous ListSessions
      call. Provide this to retrieve the subsequent page. When paginating, all
      other parameters provided to ListSessions must match the call that
      provided the page token.
    parent: Required. The resource name of the parent environment: projects/{p
      roject_number}/locations/{location_id}/lakes/{lake_id}/environment/{envi
      ronment_id}.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)