from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class BuildEnvironmentVariablesValue(_messages.Message):
    """Build environment variables that shall be available during build time.

    Messages:
      AdditionalProperty: An additional property for a
        BuildEnvironmentVariablesValue object.

    Fields:
      additionalProperties: Additional properties of type
        BuildEnvironmentVariablesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a BuildEnvironmentVariablesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)