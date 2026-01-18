from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ListAppGroupAppsResponse(_messages.Message):
    """Response for ListAppGroupApps

  Fields:
    appGroupApps: List of AppGroup apps and their credentials.
    nextPageToken: Token that can be sent as `next_page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
  """
    appGroupApps = _messages.MessageField('GoogleCloudApigeeV1AppGroupApp', 1, repeated=True)
    nextPageToken = _messages.StringField(2)