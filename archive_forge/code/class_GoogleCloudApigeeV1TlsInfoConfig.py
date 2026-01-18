from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1TlsInfoConfig(_messages.Message):
    """A GoogleCloudApigeeV1TlsInfoConfig object.

  Fields:
    ciphers: List of ciphers that are granted access.
    clientAuthEnabled: Flag that specifies whether client-side authentication
      is enabled for the target server. Enables two-way TLS.
    commonName: Common name to validate the target server against.
    enabled: Flag that specifies whether one-way TLS is enabled. Set to `true`
      to enable one-way TLS.
    ignoreValidationErrors: Flag that specifies whether to ignore TLS
      certificate validation errors. Set to `true` to ignore errors.
    keyAlias: Name of the alias used for client-side authentication in the
      following format: `organizations/{org}/environments/{env}/keystores/{key
      store}/aliases/{alias}`
    keyAliasReference: Reference name and alias pair to use for client-side
      authentication.
    protocols: List of TLS protocols that are granted access.
    trustStore: Name of the keystore or keystore reference containing trusted
      certificates for the server in the following format:
      `organizations/{org}/environments/{env}/keystores/{keystore}` or
      `organizations/{org}/environments/{env}/references/{reference}`
  """
    ciphers = _messages.StringField(1, repeated=True)
    clientAuthEnabled = _messages.BooleanField(2)
    commonName = _messages.MessageField('GoogleCloudApigeeV1CommonNameConfig', 3)
    enabled = _messages.BooleanField(4)
    ignoreValidationErrors = _messages.BooleanField(5)
    keyAlias = _messages.StringField(6)
    keyAliasReference = _messages.MessageField('GoogleCloudApigeeV1KeyAliasReference', 7)
    protocols = _messages.StringField(8, repeated=True)
    trustStore = _messages.StringField(9)