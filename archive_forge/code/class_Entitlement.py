from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Entitlement(_messages.Message):
    """Proto representing the access that a user has to a specific
  feature/service. NextId: 3.

  Enums:
    EntitlementStateValueValuesEnum: The current state of user's accessibility
      to a feature/benefit.
    TypeValueValuesEnum: An enum that represents the type of this entitlement.

  Fields:
    entitlementState: The current state of user's accessibility to a
      feature/benefit.
    type: An enum that represents the type of this entitlement.
  """

    class EntitlementStateValueValuesEnum(_messages.Enum):
        """The current state of user's accessibility to a feature/benefit.

    Values:
      ENTITLEMENT_STATE_UNSPECIFIED: <no description>
      ENTITLED: User is entitled to a feature/benefit, but whether it has been
        successfully provisioned is decided by provisioning state.
      REVOKED: User is entitled to a feature/benefit, but it was requested to
        be revoked. Whether the revoke has been successful is decided by
        provisioning state.
    """
        ENTITLEMENT_STATE_UNSPECIFIED = 0
        ENTITLED = 1
        REVOKED = 2

    class TypeValueValuesEnum(_messages.Enum):
        """An enum that represents the type of this entitlement.

    Values:
      ENTITLEMENT_TYPE_UNSPECIFIED: <no description>
      DUET_AI: The root entitlement representing Duet AI package ownership.
      GEMINI: The root entitlement representing Gemini package ownership.
    """
        ENTITLEMENT_TYPE_UNSPECIFIED = 0
        DUET_AI = 1
        GEMINI = 2
    entitlementState = _messages.EnumField('EntitlementStateValueValuesEnum', 1)
    type = _messages.EnumField('TypeValueValuesEnum', 2)