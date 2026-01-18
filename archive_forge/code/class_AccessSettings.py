from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccessSettings(_messages.Message):
    """Access related settings for IAP protected apps.

  Fields:
    allowedDomainsSettings: Settings to configure and enable allowed domains.
    corsSettings: Configuration to allow cross-origin requests via IAP.
    gcipSettings: GCIP claims and endpoint configurations for 3p identity
      providers.
    oauthSettings: Settings to configure IAP's OAuth behavior.
    policyDelegationSettings: Settings to configure Policy delegation for apps
      hosted in tenant projects. INTERNAL_ONLY.
    reauthSettings: Settings to configure reauthentication policies in IAP.
  """
    allowedDomainsSettings = _messages.MessageField('AllowedDomainsSettings', 1)
    corsSettings = _messages.MessageField('CorsSettings', 2)
    gcipSettings = _messages.MessageField('GcipSettings', 3)
    oauthSettings = _messages.MessageField('OAuthSettings', 4)
    policyDelegationSettings = _messages.MessageField('PolicyDelegationSettings', 5)
    reauthSettings = _messages.MessageField('ReauthSettings', 6)