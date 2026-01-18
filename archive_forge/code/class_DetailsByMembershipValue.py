from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class DetailsByMembershipValue(_messages.Message):
    """FeatureState for each Membership. Keys are the fully-qualified
    Membership name in the format
    `projects/{NUMBER}/locations/*/memberships/*`.

    Messages:
      AdditionalProperty: An additional property for a
        DetailsByMembershipValue object.

    Fields:
      additionalProperties: Additional properties of type
        DetailsByMembershipValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a DetailsByMembershipValue object.

      Fields:
        key: Name of the additional property.
        value: A FeatureStateDetails attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('FeatureStateDetails', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)