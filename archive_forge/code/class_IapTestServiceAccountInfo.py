from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class IapTestServiceAccountInfo(_messages.Message):
    """Describes authentication configuration when Web-Security-Scanner service
  account is added in Identity-Aware-Proxy (IAP) access policies.

  Fields:
    targetAudienceClientId: Required. Describes OAuth2 Client ID of resources
      protected by Identity-Aware-Proxy(IAP).
  """
    targetAudienceClientId = _messages.StringField(1)