from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArtifactregistryProjectsLocationsRepositoriesRulesCreateRequest(_messages.Message):
    """A ArtifactregistryProjectsLocationsRepositoriesRulesCreateRequest
  object.

  Fields:
    googleDevtoolsArtifactregistryV1Rule: A
      GoogleDevtoolsArtifactregistryV1Rule resource to be passed as the
      request body.
    parent: Required. The name of the parent resource where the rule will be
      created.
    ruleId: The rule id to use for this repository.
  """
    googleDevtoolsArtifactregistryV1Rule = _messages.MessageField('GoogleDevtoolsArtifactregistryV1Rule', 1)
    parent = _messages.StringField(2, required=True)
    ruleId = _messages.StringField(3)