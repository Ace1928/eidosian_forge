from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class MembershipStatesValue(_messages.Message):
    """Output only. Membership-specific Feature status. If this Feature does
    report any per-Membership status, this field may be unused. The keys
    indicate which Membership the state is for, in the form:
    `projects/{p}/locations/{l}/memberships/{m}` Where {p} is the project
    number, {l} is a valid location and {m} is a valid Membership in this
    project at that location. {p} MUST match the Feature's project number.

    Messages:
      AdditionalProperty: An additional property for a MembershipStatesValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        MembershipStatesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a MembershipStatesValue object.

      Fields:
        key: Name of the additional property.
        value: A MembershipFeatureState attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('MembershipFeatureState', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)