from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2CreateStoredInfoTypeRequest(_messages.Message):
    """Request message for CreateStoredInfoType.

  Fields:
    config: Required. Configuration of the storedInfoType to create.
    locationId: Deprecated. This field has no effect.
    storedInfoTypeId: The storedInfoType ID can contain uppercase and
      lowercase letters, numbers, and hyphens; that is, it must match the
      regular expression: `[a-zA-Z\\d-_]+`. The maximum length is 100
      characters. Can be empty to allow the system to generate one.
  """
    config = _messages.MessageField('GooglePrivacyDlpV2StoredInfoTypeConfig', 1)
    locationId = _messages.StringField(2)
    storedInfoTypeId = _messages.StringField(3)