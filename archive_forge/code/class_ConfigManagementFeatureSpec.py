from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigManagementFeatureSpec(_messages.Message):
    """Spec for Anthos Config Management (ACM).

  Messages:
    MembershipConfigsValue: Map of Membership IDs to individual configs.

  Fields:
    membershipConfigs: Map of Membership IDs to individual configs.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MembershipConfigsValue(_messages.Message):
        """Map of Membership IDs to individual configs.

    Messages:
      AdditionalProperty: An additional property for a MembershipConfigsValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        MembershipConfigsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MembershipConfigsValue object.

      Fields:
        key: Name of the additional property.
        value: A MembershipConfig attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('MembershipConfig', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    membershipConfigs = _messages.MessageField('MembershipConfigsValue', 1)