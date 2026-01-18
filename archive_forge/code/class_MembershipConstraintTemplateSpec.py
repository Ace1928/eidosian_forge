from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MembershipConstraintTemplateSpec(_messages.Message):
    """The spec defining this constraint template. See https://github.com/open-
  policy-agent/frameworks/tree/master/constraint#what-is-a-constraint-
  template.

  Messages:
    PropertiesValue: spec.crd.spec.validation.openAPIV3Schema.

  Fields:
    constraintKind: spec.crd.spec.names.kind.
    properties: spec.crd.spec.validation.openAPIV3Schema.
    targets: spec.targets. Use a list of targets to account for multi-target
      templates.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PropertiesValue(_messages.Message):
        """spec.crd.spec.validation.openAPIV3Schema.

    Messages:
      AdditionalProperty: An additional property for a PropertiesValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a PropertiesValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    constraintKind = _messages.StringField(1)
    properties = _messages.MessageField('PropertiesValue', 2)
    targets = _messages.MessageField('Target', 3, repeated=True)