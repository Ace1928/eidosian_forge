from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkforcePoolProviderKey(_messages.Message):
    """Represents a public key configuration for a Workforce Pool Provider. The
  key can be configured in your identity provider to encrypt SAML assertions.
  Google holds the corresponding private key, which it uses to decrypt
  encrypted tokens.

  Enums:
    StateValueValuesEnum: Output only. The state of the key.
    UseValueValuesEnum: Required. The purpose of the key.

  Fields:
    expireTime: Output only. The time after which the key will be permanently
      deleted and cannot be recovered. Note that the key may get purged before
      this time if the total limit of keys per provider is exceeded.
    keyData: Immutable. Public half of the asymmetric key.
    name: Output only. The resource name of the key.
    state: Output only. The state of the key.
    use: Required. The purpose of the key.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of the key.

    Values:
      STATE_UNSPECIFIED: State unspecified.
      ACTIVE: The key is active.
      DELETED: The key is soft-deleted. Soft-deleted keys are permanently
        deleted after approximately 30 days. You can restore a soft-deleted
        key using UndeleteWorkforcePoolProviderKey.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        DELETED = 2

    class UseValueValuesEnum(_messages.Enum):
        """Required. The purpose of the key.

    Values:
      KEY_USE_UNSPECIFIED: KeyUse unspecified.
      ENCRYPTION: The key is used for encryption.
    """
        KEY_USE_UNSPECIFIED = 0
        ENCRYPTION = 1
    expireTime = _messages.StringField(1)
    keyData = _messages.MessageField('KeyData', 2)
    name = _messages.StringField(3)
    state = _messages.EnumField('StateValueValuesEnum', 4)
    use = _messages.EnumField('UseValueValuesEnum', 5)