from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicyControllerFeatureSpec(_messages.Message):
    """Spec for Policy Controller.

  Messages:
    MembershipSpecsValue: Map of Membership IDs to individual specs.

  Fields:
    membershipSpecs: Map of Membership IDs to individual specs.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MembershipSpecsValue(_messages.Message):
        """Map of Membership IDs to individual specs.

    Messages:
      AdditionalProperty: An additional property for a MembershipSpecsValue
        object.

    Fields:
      additionalProperties: Additional properties of type MembershipSpecsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MembershipSpecsValue object.

      Fields:
        key: Name of the additional property.
        value: A PolicyControllerMembershipSpec attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('PolicyControllerMembershipSpec', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    membershipSpecs = _messages.MessageField('MembershipSpecsValue', 1)