from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class TargetArtifactsValue(_messages.Message):
    """Output only. Map from target ID to the target artifacts created during
    the render operation.

    Messages:
      AdditionalProperty: An additional property for a TargetArtifactsValue
        object.

    Fields:
      additionalProperties: Additional properties of type TargetArtifactsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a TargetArtifactsValue object.

      Fields:
        key: Name of the additional property.
        value: A TargetArtifact attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('TargetArtifact', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)