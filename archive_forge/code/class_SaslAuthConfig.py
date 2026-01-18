from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SaslAuthConfig(_messages.Message):
    """SASL/Plain or SASL/SCRAM mechanism configuration.

  Enums:
    MechanismValueValuesEnum:

  Fields:
    mechanism: A MechanismValueValuesEnum attribute.
    passwordSecret: Required. The password for the authentication identity may
      be loaded from Secret Manager.
    username: Required. The SASL authentication identity (username).
  """

    class MechanismValueValuesEnum(_messages.Enum):
        """MechanismValueValuesEnum enum type.

    Values:
      AUTH_MECHANISM_UNSPECIFIED: <no description>
      PLAIN: <no description>
      SHA_256: <no description>
      SHA_512: <no description>
    """
        AUTH_MECHANISM_UNSPECIFIED = 0
        PLAIN = 1
        SHA_256 = 2
        SHA_512 = 3
    mechanism = _messages.EnumField('MechanismValueValuesEnum', 1)
    passwordSecret = _messages.StringField(2)
    username = _messages.StringField(3)