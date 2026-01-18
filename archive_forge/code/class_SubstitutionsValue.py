from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class SubstitutionsValue(_messages.Message):
    """Substitutions to use in a triggered build. Should only be used with
    RunBuildTrigger

    Messages:
      AdditionalProperty: An additional property for a SubstitutionsValue
        object.

    Fields:
      additionalProperties: Additional properties of type SubstitutionsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a SubstitutionsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)