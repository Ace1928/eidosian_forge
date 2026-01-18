from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EventingConfig(_messages.Message):
    """Eventing Configuration of a connection

  Fields:
    additionalVariables: Additional eventing related field values
    authConfig: Auth details for the webhook adapter.
    encryptionKey: Encryption key (can be either Google managed or CMEK).
    enrichmentEnabled: Enrichment Enabled.
    registrationDestinationConfig: Registration endpoint for auto
      regsitration.
  """
    additionalVariables = _messages.MessageField('ConfigVariable', 1, repeated=True)
    authConfig = _messages.MessageField('AuthConfig', 2)
    encryptionKey = _messages.MessageField('ConfigVariable', 3)
    enrichmentEnabled = _messages.BooleanField(4)
    registrationDestinationConfig = _messages.MessageField('DestinationConfig', 5)