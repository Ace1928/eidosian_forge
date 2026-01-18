from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeaturestoresEntityTypesDeleteRequest(_messages.Message):
    """A AiplatformProjectsLocationsFeaturestoresEntityTypesDeleteRequest
  object.

  Fields:
    force: If set to true, any Features for this EntityType will also be
      deleted. (Otherwise, the request will only work if the EntityType has no
      Features.)
    name: Required. The name of the EntityType to be deleted. Format: `project
      s/{project}/locations/{location}/featurestores/{featurestore}/entityType
      s/{entity_type}`
  """
    force = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)