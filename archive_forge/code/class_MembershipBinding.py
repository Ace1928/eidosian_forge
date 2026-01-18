from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MembershipBinding(_messages.Message):
    """MembershipBinding is a subresource of a Membership, representing what
  Fleet Scopes (or other, future Fleet resources) a Membership is bound to.

  Messages:
    LabelsValue: Optional. Labels for this MembershipBinding.

  Fields:
    createTime: Output only. When the membership binding was created.
    deleteTime: Output only. When the membership binding was deleted.
    labels: Optional. Labels for this MembershipBinding.
    name: The resource name for the membershipbinding itself `projects/{projec
      t}/locations/{location}/memberships/{membership}/bindings/{membershipbin
      ding}`
    scope: A Scope resource name in the format
      `projects/*/locations/*/scopes/*`.
    state: Output only. State of the membership binding resource.
    uid: Output only. Google-generated UUID for this resource. This is unique
      across all membershipbinding resources. If a membershipbinding resource
      is deleted and another resource with the same name is created, it gets a
      different uid.
    updateTime: Output only. When the membership binding was last updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Labels for this MembershipBinding.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    createTime = _messages.StringField(1)
    deleteTime = _messages.StringField(2)
    labels = _messages.MessageField('LabelsValue', 3)
    name = _messages.StringField(4)
    scope = _messages.StringField(5)
    state = _messages.MessageField('MembershipBindingLifecycleState', 6)
    uid = _messages.StringField(7)
    updateTime = _messages.StringField(8)