from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2UpdateStoredInfoTypeRequest(_messages.Message):
    """Request message for UpdateStoredInfoType.

  Fields:
    config: Updated configuration for the storedInfoType. If not provided, a
      new version of the storedInfoType will be created with the existing
      configuration.
    updateMask: Mask to control which fields get updated.
  """
    config = _messages.MessageField('GooglePrivacyDlpV2StoredInfoTypeConfig', 1)
    updateMask = _messages.StringField(2)