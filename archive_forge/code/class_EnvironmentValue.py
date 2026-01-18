from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class EnvironmentValue(_messages.Message):
    """Any other builder-controlled inputs necessary for correctly evaluating
    the recipe. Usually only needed for reproducing the build but not
    evaluated as part of policy. Depending on the recipe Type, the structure
    may be different.

    Messages:
      AdditionalProperty: An additional property for a EnvironmentValue
        object.

    Fields:
      additionalProperties: Properties of the object. Contains field @type
        with type URL.
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a EnvironmentValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('extra_types.JsonValue', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)