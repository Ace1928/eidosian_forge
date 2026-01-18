from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeregistryProjectsLocationsArtifactsCreateRequest(_messages.Message):
    """A ApigeeregistryProjectsLocationsArtifactsCreateRequest object.

  Fields:
    artifact: A Artifact resource to be passed as the request body.
    artifactId: Required. The ID to use for the artifact, which will become
      the final component of the artifact's resource name. This value should
      be 4-63 characters, and valid characters are /a-z-/. Following AIP-162,
      IDs must not have the form of a UUID.
    parent: Required. The parent, which owns this collection of artifacts.
      Format: `{parent}`
  """
    artifact = _messages.MessageField('Artifact', 1)
    artifactId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)