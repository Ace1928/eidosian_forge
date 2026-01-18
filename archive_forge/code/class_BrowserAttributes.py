from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BrowserAttributes(_messages.Message):
    """Contains information about browser profiles reported by the Endpoint
  Verification extension.

  Fields:
    chromeBrowserInfo: Represents the current state of the [Chrome browser
      attributes](https://cloud.google.com/access-context-
      manager/docs/browser-attributes) sent by the Endpoint Verification
      extension.
    chromeProfileId: Chrome profile ID that is exposed by the Chrome API. It
      is unique for each device.
    lastProfileSyncTime: Timestamp in milliseconds since Epoch when the
      profile/gcm id was last synced.
  """
    chromeBrowserInfo = _messages.MessageField('BrowserInfo', 1)
    chromeProfileId = _messages.StringField(2)
    lastProfileSyncTime = _messages.StringField(3)