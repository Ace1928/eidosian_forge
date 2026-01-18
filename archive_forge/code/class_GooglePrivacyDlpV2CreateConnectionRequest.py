from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2CreateConnectionRequest(_messages.Message):
    """Request message for CreateConnection.

  Fields:
    connection: Required. The connection resource.
  """
    connection = _messages.MessageField('GooglePrivacyDlpV2Connection', 1)