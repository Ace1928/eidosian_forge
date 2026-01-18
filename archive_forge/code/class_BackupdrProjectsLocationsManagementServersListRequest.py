from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackupdrProjectsLocationsManagementServersListRequest(_messages.Message):
    """A BackupdrProjectsLocationsManagementServersListRequest object.

  Fields:
    filter: Optional. Filtering results.
    orderBy: Optional. Hint for how to order the results.
    pageSize: Optional. Requested page size. Server may return fewer items
      than requested. If unspecified, server will pick an appropriate default.
    pageToken: Optional. A token identifying a page of results the server
      should return.
    parent: Required. The project and location for which to retrieve
      management servers information, in the format
      `projects/{project_id}/locations/{location}`. In Cloud BackupDR,
      locations map to GCP regions, for example **us-central1**. To retrieve
      management servers for all locations, use "-" for the `{location}`
      value.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)