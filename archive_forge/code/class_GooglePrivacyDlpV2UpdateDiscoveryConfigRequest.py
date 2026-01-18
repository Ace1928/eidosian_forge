from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2UpdateDiscoveryConfigRequest(_messages.Message):
    """Request message for UpdateDiscoveryConfig.

  Fields:
    discoveryConfig: Required. New DiscoveryConfig value.
    updateMask: Mask to control which fields get updated.
  """
    discoveryConfig = _messages.MessageField('GooglePrivacyDlpV2DiscoveryConfig', 1)
    updateMask = _messages.StringField(2)