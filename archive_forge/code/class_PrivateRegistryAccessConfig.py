from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrivateRegistryAccessConfig(_messages.Message):
    """PrivateRegistryAccessConfig contains access configuration for private
  container registries.

  Fields:
    certificateAuthorityDomainConfig: Private registry access configuration.
    enabled: Private registry access is enabled.
  """
    certificateAuthorityDomainConfig = _messages.MessageField('CertificateAuthorityDomainConfig', 1, repeated=True)
    enabled = _messages.BooleanField(2)