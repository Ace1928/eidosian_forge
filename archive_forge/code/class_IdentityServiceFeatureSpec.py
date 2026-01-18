from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IdentityServiceFeatureSpec(_messages.Message):
    """Spec for Annthos Identity Service.

  Messages:
    MemberConfigsValue: A map between member ids to their configurations. The
      ID needs to be the full path to the membership e.g.,
      /projects/p/locations/l/memberships/m.

  Fields:
    memberConfigs: A map between member ids to their configurations. The ID
      needs to be the full path to the membership e.g.,
      /projects/p/locations/l/memberships/m.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MemberConfigsValue(_messages.Message):
        """A map between member ids to their configurations. The ID needs to be
    the full path to the membership e.g.,
    /projects/p/locations/l/memberships/m.

    Messages:
      AdditionalProperty: An additional property for a MemberConfigsValue
        object.

    Fields:
      additionalProperties: Additional properties of type MemberConfigsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MemberConfigsValue object.

      Fields:
        key: Name of the additional property.
        value: A MemberConfig attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('MemberConfig', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    memberConfigs = _messages.MessageField('MemberConfigsValue', 1)