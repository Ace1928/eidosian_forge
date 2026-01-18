from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingFoldersUpdateSettingsRequest(_messages.Message):
    """A LoggingFoldersUpdateSettingsRequest object.

  Fields:
    name: Required. The resource name for the settings to update.
      "organizations/[ORGANIZATION_ID]/settings" For
      example:"organizations/12345/settings"
    settings: A Settings resource to be passed as the request body.
    updateMask: Optional. Field mask identifying which fields from settings
      should be updated. A field will be overwritten if and only if it is in
      the update mask. Output only fields cannot be updated.See FieldMask for
      more information.For example: "updateMask=kmsKeyName"
  """
    name = _messages.StringField(1, required=True)
    settings = _messages.MessageField('Settings', 2)
    updateMask = _messages.StringField(3)