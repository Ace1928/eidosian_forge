from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class MembershipConfigsValue(_messages.Message):
    """Map of Membership resource name to individual configs. Membership
    resource names are in the format of "projects/P/locations/L/memberships/M"

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
        value: A NamespaceActuationMembershipSpec attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('NamespaceActuationMembershipSpec', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)