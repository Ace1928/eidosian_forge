from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetArtifact(_messages.Message):
    """The artifacts produced by a target render operation.

  Messages:
    PhaseArtifactsValue: Output only. Map from the phase ID to the phase
      artifacts for the `Target`.

  Fields:
    artifactUri: Output only. URI of a directory containing the artifacts.
      This contains deployment configuration used by Skaffold during a
      rollout, and all paths are relative to this location.
    manifestPath: Output only. File path of the rendered manifest relative to
      the URI.
    phaseArtifacts: Output only. Map from the phase ID to the phase artifacts
      for the `Target`.
    skaffoldConfigPath: Output only. File path of the resolved Skaffold
      configuration relative to the URI.
  """

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
    artifactUri = _messages.StringField(1)
    manifestPath = _messages.StringField(2)
    phaseArtifacts = _messages.MessageField('PhaseArtifactsValue', 3)
    skaffoldConfigPath = _messages.StringField(4)