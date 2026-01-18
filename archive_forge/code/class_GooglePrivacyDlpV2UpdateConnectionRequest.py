from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2UpdateConnectionRequest(_messages.Message):
    """Request message for UpdateConnection.

  Fields:
    connection: Required. The connection with new values for the relevant
      fields.
    updateMask: Optional. Mask to control which fields get updated.
  """
    connection = _messages.MessageField('GooglePrivacyDlpV2Connection', 1)
    updateMask = _messages.StringField(2)