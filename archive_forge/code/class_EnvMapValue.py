from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class EnvMapValue(_messages.Message):
    """Map of environment variable name to its encrypted value. Secret
    environment variables must be unique across all of a build's secrets, and
    must be used by at least one build step. Values can be at most 64 KB in
    size. There can be at most 100 secret values across all of a build's
    secrets.

    Messages:
      AdditionalProperty: An additional property for a EnvMapValue object.

    Fields:
      additionalProperties: Additional properties of type EnvMapValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a EnvMapValue object.

      Fields:
        key: Name of the additional property.
        value: A byte attribute.
      """
        key = _messages.StringField(1)
        value = _messages.BytesField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)