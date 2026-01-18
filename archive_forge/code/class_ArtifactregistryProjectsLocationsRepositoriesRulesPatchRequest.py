from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArtifactregistryProjectsLocationsRepositoriesRulesPatchRequest(_messages.Message):
    """A ArtifactregistryProjectsLocationsRepositoriesRulesPatchRequest object.

  Fields:
    googleDevtoolsArtifactregistryV1Rule: A
      GoogleDevtoolsArtifactregistryV1Rule resource to be passed as the
      request body.
    name: The name of the rule, for example: "projects/p1/locations/us-
      central1/repositories/repo1/rules/rule1".
    updateMask: The update mask applies to the resource. For the `FieldMask`
      definition, see https://developers.google.com/protocol-
      buffers/docs/reference/google.protobuf#fieldmask
  """
    googleDevtoolsArtifactregistryV1Rule = _messages.MessageField('GoogleDevtoolsArtifactregistryV1Rule', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)