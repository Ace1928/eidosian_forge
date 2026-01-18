from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2CreateDiscoveryConfigRequest(_messages.Message):
    """Request message for CreateDiscoveryConfig.

  Fields:
    configId: The config ID can contain uppercase and lowercase letters,
      numbers, and hyphens; that is, it must match the regular expression:
      `[a-zA-Z\\d-_]+`. The maximum length is 100 characters. Can be empty to
      allow the system to generate one.
    discoveryConfig: Required. The DiscoveryConfig to create.
  """
    configId = _messages.StringField(1)
    discoveryConfig = _messages.MessageField('GooglePrivacyDlpV2DiscoveryConfig', 2)