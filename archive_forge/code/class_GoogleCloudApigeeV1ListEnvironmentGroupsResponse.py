from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ListEnvironmentGroupsResponse(_messages.Message):
    """Response for ListEnvironmentGroups.

  Fields:
    environmentGroups: EnvironmentGroups in the specified organization.
    nextPageToken: Page token that you can include in a ListEnvironmentGroups
      request to retrieve the next page. If omitted, no subsequent pages
      exist.
  """
    environmentGroups = _messages.MessageField('GoogleCloudApigeeV1EnvironmentGroup', 1, repeated=True)
    nextPageToken = _messages.StringField(2)