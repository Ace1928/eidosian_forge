from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceMeshFeatureSpec(_messages.Message):
    """ServiceMeshFeatureSpec contains the input for the service mesh feature.

  Messages:
    MembershipSpecsValue: Optional. Map from full path to the membership, to
      its individual config.

  Fields:
    fleetDefaultMemberConfig: Optional. The default fleet configuration to be
      applied to all memberships.
    membershipSpecs: Optional. Map from full path to the membership, to its
      individual config.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MembershipSpecsValue(_messages.Message):
        """Optional. Map from full path to the membership, to its individual
    config.

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
        value: A ServiceMeshMembershipSpec attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('ServiceMeshMembershipSpec', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    fleetDefaultMemberConfig = _messages.MessageField('ServiceMeshMembershipSpec', 1)
    membershipSpecs = _messages.MessageField('MembershipSpecsValue', 2)