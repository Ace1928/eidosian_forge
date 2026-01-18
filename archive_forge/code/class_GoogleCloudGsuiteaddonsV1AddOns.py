from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGsuiteaddonsV1AddOns(_messages.Message):
    """A Google Workspace Add-on configuration.

  Fields:
    calendar: Calendar add-on configuration.
    common: Configuration that is common across all Google Workspace Add-ons.
    docs: Docs add-on configuration.
    drive: Drive add-on configuration.
    gmail: Gmail add-on configuration.
    httpOptions: Options for sending requests to add-on HTTP endpoints
    sheets: Sheets add-on configuration.
    slides: Slides add-on configuration.
  """
    calendar = _messages.MessageField('GoogleAppsScriptTypeCalendarCalendarAddOnManifest', 1)
    common = _messages.MessageField('GoogleAppsScriptTypeCommonAddOnManifest', 2)
    docs = _messages.MessageField('GoogleAppsScriptTypeDocsDocsAddOnManifest', 3)
    drive = _messages.MessageField('GoogleAppsScriptTypeDriveDriveAddOnManifest', 4)
    gmail = _messages.MessageField('GoogleAppsScriptTypeGmailGmailAddOnManifest', 5)
    httpOptions = _messages.MessageField('GoogleAppsScriptTypeHttpOptions', 6)
    sheets = _messages.MessageField('GoogleAppsScriptTypeSheetsSheetsAddOnManifest', 7)
    slides = _messages.MessageField('GoogleAppsScriptTypeSlidesSlidesAddOnManifest', 8)