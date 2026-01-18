from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListWorkforcePoolInstalledAppsResponse(_messages.Message):
    """Response message for ListWorkforcePoolInstalledApps.

  Fields:
    nextPageToken: Optional. A token, which can be sent as `page_token` to
      retrieve the next page. If this field is omitted, there are no
      subsequent pages.
    workforcePoolInstalledApps: Output only. A list of workforce pool
      installed apps.
  """
    nextPageToken = _messages.StringField(1)
    workforcePoolInstalledApps = _messages.MessageField('WorkforcePoolInstalledApp', 2, repeated=True)