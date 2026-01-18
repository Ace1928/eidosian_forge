from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class PhaseArtifactsValue(_messages.Message):
    """Output only. Map from the phase ID to the phase artifacts for the
    `Target`.

    Messages:
      AdditionalProperty: An additional property for a PhaseArtifactsValue
        object.

    Fields:
      additionalProperties: Additional properties of type PhaseArtifactsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a PhaseArtifactsValue object.

      Fields:
        key: Name of the additional property.
        value: A PhaseArtifact attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('PhaseArtifact', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)